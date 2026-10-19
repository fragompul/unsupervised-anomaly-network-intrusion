import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Bloom, EffectComposer } from '@react-three/postprocessing'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { nlogp, pValue } from '../../lib/data'
import { categoryColor, hexToRgb, ramp } from '../../lib/palette'
import type { ColorBy } from '../../lib/store'
import type { DatasetData } from '../../lib/types'

export interface CloudProps {
  ds: DatasetData
  embedding: 'pca' | 'umap'
  colorBy: ColorBy
  detectorIndex: number
  alpha: number
  hiddenCategories: string[]
  selected: number | null
  onSelect?: (i: number | null) => void
  onHover?: (i: number | null, x: number, y: number) => void
  autoRotate?: boolean
  rotateSpeed?: number
  pointScale?: number
  ambient?: boolean // hero background: no picking, softer, no flagged emphasis
  bloom?: boolean
}

const VERT = /* glsl */ `
  attribute vec3 positionB;
  attribute vec3 aColor;
  attribute float aSize;
  attribute float aAlpha;
  uniform float uMix;
  uniform float uScale;
  uniform float uTime;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec3 p = mix(position, positionB, uMix);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = clamp(1.0 - (-mv.z - 1.6) / 3.2, 0.25, 1.0);
    gl_PointSize = aSize * uScale / -mv.z * (0.7 + 0.3 * depth);
    vColor = aColor;
    vAlpha = aAlpha * (0.45 + 0.55 * depth);
  }
`
const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    float core = smoothstep(0.5, 0.0, d);
    float a = pow(core, 1.35) * vAlpha;
    gl_FragColor = vec4(vColor * (0.55 + 0.45 * core), a);
  }
`
const RING_FRAG = /* glsl */ `
  uniform float uTime;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    float pulse = 0.5 + 0.5 * sin(uTime * 3.0);
    float ring = smoothstep(0.5, 0.46, d) * smoothstep(0.36 + 0.03 * pulse, 0.42, d);
    if (ring < 0.01) discard;
    gl_FragColor = vec4(1.0, 1.0, 1.0, ring);
  }
`
const RING_VERT = /* glsl */ `
  uniform float uPx;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uPx;
  }
`

function buildAppearance(p: CloudProps) {
  const { ds, colorBy, detectorIndex, alpha, hiddenCategories, ambient } = p
  const n = ds.n
  const colors = new Float32Array(n * 3)
  const sizes = new Float32Array(n)
  const alphas = new Float32Array(n)
  const cats = ds.meta.categories
  const catRgb = cats.map((c, i) => hexToRgb(categoryColor(c, i)).map((v) => v / 255))
  const hidden = new Set(hiddenCategories.map((c) => cats.indexOf(c)))
  const normalIdx = cats.indexOf('normal')
  const maxNl = 5
  for (let i = 0; i < n; i++) {
    const cat = ds.category[i]
    const isAttack = ds.attack[i] === 1
    const nl = nlogp(ds, detectorIndex, i)
    const flagged = pValue(nl) <= alpha
    let rgb: number[]
    if (colorBy === 'category') rgb = catRgb[cat]
    else if (colorBy === 'truth') rgb = isAttack ? [1, 0.36, 0.5] : [0.33, 0.42, 0.62]
    else rgb = ramp(Math.min(1, nl / maxNl)).map((v) => v / 255)
    colors[i * 3] = rgb[0]
    colors[i * 3 + 1] = rgb[1]
    colors[i * 3 + 2] = rgb[2]
    if (hidden.has(cat)) {
      sizes[i] = 0
      alphas[i] = 0
      continue
    }
    const isNormal = cat === normalIdx
    if (ambient) {
      sizes[i] = isNormal ? 6 : 9
      alphas[i] = isNormal ? 0.5 : 0.9
    } else if (colorBy === 'score') {
      sizes[i] = 6 + 14 * Math.min(1, nl / maxNl)
      alphas[i] = 0.35 + 0.65 * Math.min(1, nl / (maxNl * 0.6))
    } else {
      sizes[i] = (isNormal ? 7 : 10) * (flagged ? 1.55 : 1)
      alphas[i] = flagged ? 1 : isNormal ? 0.36 : 0.6
    }
  }
  return { colors, sizes, alphas }
}

function Cloud(p: CloudProps) {
  const { ds, embedding, selected, onSelect, onHover, ambient } = p
  const { gl, camera, size, invalidate } = useThree()
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    const src = ds.embeddings[embedding] ?? ds.embeddings.pca!
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(src), 3).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('positionB', new THREE.BufferAttribute(new Float32Array(src), 3).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(ds.n * 3), 3).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(ds.n), 1).setUsage(THREE.DynamicDrawUsage))
    g.setAttribute('aAlpha', new THREE.BufferAttribute(new Float32Array(ds.n), 1).setUsage(THREE.DynamicDrawUsage))
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 2)
    return g
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ds])

  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: { uMix: { value: 0 }, uScale: { value: 1 }, uTime: { value: 0 } },
      }),
    [],
  )
  const ringMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: RING_VERT,
        fragmentShader: RING_FRAG,
        transparent: true,
        depthTest: false,
        uniforms: { uTime: { value: 0 }, uPx: { value: 30 } },
      }),
    [],
  )
  const ringGeom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3))
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 10)
    return g
  }, [])

  // current displayed positions and morph state
  const shown = useRef<Float32Array>(new Float32Array(ds.embeddings[embedding] ?? ds.embeddings.pca!))
  const morph = useRef({ active: false, t0: 0 })
  const lastEmb = useRef(embedding)

  useEffect(() => {
    if (lastEmb.current === embedding) return
    lastEmb.current = embedding
    const target = ds.embeddings[embedding]
    if (!target) return
    const a = geom.getAttribute('position') as THREE.BufferAttribute
    const b = geom.getAttribute('positionB') as THREE.BufferAttribute
    a.array.set(shown.current)
    a.needsUpdate = true
    b.array.set(target)
    b.needsUpdate = true
    morph.current = { active: true, t0: performance.now() }
    invalidate()
  }, [embedding, ds, geom, invalidate])

  const appearance = useMemo(
    () => buildAppearance(p),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ds, p.colorBy, p.detectorIndex, p.alpha, p.hiddenCategories, p.ambient],
  )
  useEffect(() => {
    ;(geom.getAttribute('aColor') as THREE.BufferAttribute).array.set(appearance.colors)
    ;(geom.getAttribute('aSize') as THREE.BufferAttribute).array.set(appearance.sizes)
    ;(geom.getAttribute('aAlpha') as THREE.BufferAttribute).array.set(appearance.alphas)
    for (const k of ['aColor', 'aSize', 'aAlpha']) (geom.getAttribute(k) as THREE.BufferAttribute).needsUpdate = true
    invalidate()
  }, [appearance, geom, invalidate])

  const group = useRef<THREE.Group>(null)

  useFrame((state) => {
    const dpr = state.gl.getPixelRatio()
    const scale = (size.height / 2) * dpr * (p.pointScale ?? 1) * 0.0058
    mat.uniforms.uScale.value = scale
    ringMat.uniforms.uPx.value = 30 * dpr
    mat.uniforms.uTime.value = ringMat.uniforms.uTime.value = state.clock.elapsedTime
    if (morph.current.active) {
      const f = Math.min(1, (performance.now() - morph.current.t0) / 1100)
      const e = f < 0.5 ? 4 * f * f * f : 1 - Math.pow(-2 * f + 2, 3) / 2
      mat.uniforms.uMix.value = e
      const a = geom.getAttribute('position').array as Float32Array
      const b = geom.getAttribute('positionB').array as Float32Array
      for (let i = 0; i < a.length; i++) shown.current[i] = a[i] + (b[i] - a[i]) * e
      if (f >= 1) {
        morph.current.active = false
        mat.uniforms.uMix.value = 0
        const A = geom.getAttribute('position') as THREE.BufferAttribute
        A.array.set(b)
        A.needsUpdate = true
      }
    }
    // selection ring follows the (possibly morphing) point
    const rp = ringGeom.getAttribute('position') as THREE.BufferAttribute
    if (selected != null) {
      rp.setXYZ(0, shown.current[selected * 3], shown.current[selected * 3 + 1], shown.current[selected * 3 + 2])
      rp.needsUpdate = true
    }
  })

  // CPU picking: project all visible points, take the nearest to the pointer (front-most on ties)
  useEffect(() => {
    if (ambient || (!onSelect && !onHover)) return
    const el = gl.domElement
    const v = new THREE.Vector3()
    let down: { x: number; y: number } | null = null
    let raf = 0
    const pick = (cx: number, cy: number): number | null => {
      const rect = el.getBoundingClientRect()
      const px = cx - rect.left
      const py = cy - rect.top
      camera.updateMatrixWorld()
      const sizes = appearance.sizes
      let best = -1
      let bestScore = Infinity
      const arr = shown.current
      const group3 = group.current
      if (group3) group3.updateMatrixWorld()
      for (let i = 0; i < ds.n; i++) {
        if (sizes[i] === 0) continue
        v.set(arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2])
        if (group3) v.applyMatrix4(group3.matrixWorld)
        v.project(camera)
        if (v.z > 1) continue
        const sx = (v.x * 0.5 + 0.5) * rect.width
        const sy = (-v.y * 0.5 + 0.5) * rect.height
        const dx = sx - px
        const dy = sy - py
        const d2 = dx * dx + dy * dy
        if (d2 > 144) continue
        const score = d2 + v.z * 40
        if (score < bestScore) {
          bestScore = score
          best = i
        }
      }
      return best >= 0 ? best : null
    }
    const move = (e: PointerEvent) => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        const i = pick(e.clientX, e.clientY)
        el.style.cursor = i != null ? 'pointer' : 'grab'
        onHover?.(i, e.clientX, e.clientY)
      })
    }
    const leave = () => onHover?.(null, 0, 0)
    const pd = (e: PointerEvent) => (down = { x: e.clientX, y: e.clientY })
    const pu = (e: PointerEvent) => {
      if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4) onSelect?.(pick(e.clientX, e.clientY))
      down = null
    }
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerleave', leave)
    el.addEventListener('pointerdown', pd)
    el.addEventListener('pointerup', pu)
    return () => {
      cancelAnimationFrame(raf)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerleave', leave)
      el.removeEventListener('pointerdown', pd)
      el.removeEventListener('pointerup', pu)
    }
  }, [gl, camera, ds, appearance, onSelect, onHover, ambient])

  return (
    <group ref={group}>
      <points geometry={geom} material={mat} frustumCulled={false} />
      {selected != null && <points geometry={ringGeom} material={ringMat} frustumCulled={false} renderOrder={10} />}
    </group>
  )
}

function Bounds() {
  const lines = useMemo(() => {
    const g = new THREE.BoxGeometry(2, 2, 2)
    return new THREE.EdgesGeometry(g)
  }, [])
  return (
    <lineSegments geometry={lines}>
      <lineBasicMaterial color="#8b7bff" transparent opacity={0.1} />
    </lineSegments>
  )
}

export function PointCloud(props: CloudProps & { className?: string }) {
  const { className, autoRotate = true, rotateSpeed = 0.5, ambient, bloom = true } = props
  return (
    <div className={className} style={{ touchAction: 'none' }}>
      <Canvas
        key={props.ds.meta.key}
        camera={{ position: [0, 0.3, 2.65], fov: 42, near: 0.1, far: 50 }}
        dpr={[1, 2]}
        gl={{ antialias: false, alpha: true, powerPreference: 'high-performance', preserveDrawingBuffer: true }}
        frameloop="always"
      >
        {!ambient && <Bounds />}
        <Cloud {...props} />
        <OrbitControls
          enablePan={false}
          enableDamping
          dampingFactor={0.08}
          autoRotate={autoRotate}
          autoRotateSpeed={rotateSpeed}
          minDistance={1.4}
          maxDistance={6}
          enableZoom={!ambient}
          enableRotate={!ambient}
        />
        {bloom && (
          <EffectComposer multisampling={0}>
            <Bloom intensity={ambient ? 0.7 : 0.5} luminanceThreshold={ambient ? 0.3 : 0.42} luminanceSmoothing={0.4} mipmapBlur radius={0.6} />
          </EffectComposer>
        )}
      </Canvas>
    </div>
  )
}
