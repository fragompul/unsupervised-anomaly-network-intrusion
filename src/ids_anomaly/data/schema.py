"""NSL-KDD column schema and attack-category taxonomy.

NSL-KDD ships 41 engineered flow features (basic TCP/IP, content, and
time/host-based traffic statistics) plus a fine-grained attack label and a
KDD99-derived difficulty score. Only the 41 features are used for the
unsupervised methods; ``label`` and ``attack_category`` are held out and used
exclusively for evaluation, never for fitting any model.
"""

from __future__ import annotations

FEATURE_COLUMNS: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

ALL_COLUMNS: list[str] = [*FEATURE_COLUMNS, "label", "difficulty"]

CATEGORICAL_COLUMNS: list[str] = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS: list[str] = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

# Binary 0/1 columns that are technically "continuous" in the original KDD
# spec but behave as flags; kept in NUMERIC_COLUMNS but tracked separately
# since a few HPO search spaces (e.g. autoencoder input weighting) treat
# them differently from truly continuous rate/count features.
BINARY_FLAG_COLUMNS: list[str] = [
    "land",
    "logged_in",
    "root_shell",
    "su_attempted",
    "is_host_login",
    "is_guest_login",
]

# Standard 5-class grouping used across the NSL-KDD literature. NSL-KDD's
# test split deliberately contains attack types absent from train (e.g.
# ``mailbomb``, ``processtable``, ``worm``) to probe generalization to novel
# attacks -- exactly the setting unsupervised anomaly detection is meant for,
# since a model trained only on "normal" traffic statistics should flag any
# sufficiently deviant flow regardless of whether its exact label was ever seen.
ATTACK_CATEGORY_MAP: dict[str, str] = {
    "normal": "normal",
    # DoS
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "udpstorm": "dos",
    "processtable": "dos",
    "worm": "dos",
    "mailbomb": "dos",
    # Probe
    "satan": "probe",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "mscan": "probe",
    "saint": "probe",
    # R2L (remote to local)
    "guess_passwd": "r2l",
    "ftp_write": "r2l",
    "imap": "r2l",
    "phf": "r2l",
    "multihop": "r2l",
    "warezmaster": "r2l",
    "warezclient": "r2l",
    "spy": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "snmpguess": "r2l",
    "snmpgetattack": "r2l",
    "httptunnel": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    # U2R (user to root)
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "rootkit": "u2r",
    "perl": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "ps": "u2r",
}


def attack_category(label: str) -> str:
    """Map a fine-grained NSL-KDD label to its 5-class category.

    Falls back to ``"unknown_attack"`` rather than raising, since NSL-KDD's
    test split can in principle carry label spellings absent from the
    training-derived taxonomy above.
    """
    return ATTACK_CATEGORY_MAP.get(label, "unknown_attack" if label != "normal" else "normal")
