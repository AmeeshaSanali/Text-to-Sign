"""
SLSL Animation Library
Resolves sign language glosses to their approved 3D bone animation JSON data.
Only 'approved' entries in data/manifest.csv are served to the avatar.
"""

import os
import csv
from pathlib import Path
from typing import Optional, Dict, List, Any


class AnimationLibrary:
    def __init__(self, manifest_path: Optional[Path] = None, animations_dir: Optional[Path] = None):
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.manifest_path = manifest_path or (root_dir / "data" / "manifest.csv")
        self.animations_dir = animations_dir or (root_dir / "data" / "animations" / "slsl")
        self._manifest: Dict[str, Dict[str, Any]] = {}
        self.load_manifest()

    def load_manifest(self):
        """Loads and parses data/manifest.csv into memory."""
        self._manifest.clear()
        if not self.manifest_path.exists():
            return

        with open(self.manifest_path, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                gloss = row.get("gloss", "").strip().upper()
                if gloss:
                    self._manifest[gloss] = {
                        "gloss": gloss,
                        "label_in_dataset": row.get("label_in_dataset", ""),
                        "category": row.get("category", ""),
                        "language": row.get("language", "slsl"),
                        "source_video": row.get("source_video", ""),
                        "take": row.get("take", ""),
                        "signer_id": row.get("signer_id", "unknown"),
                        "mirrored": row.get("mirrored", "False").lower() == "true",
                        "license": row.get("license", "unconfirmed"),
                        "status": row.get("status", "todo").strip().lower(),
                        "notes": row.get("notes", ""),
                    }

    def is_approved(self, gloss: str) -> bool:
        """Returns True if the gloss has an approved animation ready to be served."""
        clean_gloss = gloss.strip().upper().replace("-", "_")
        entry = self._manifest.get(clean_gloss)
        if not entry:
            return False
        if entry["status"] != "approved":
            return False
        # Confirm JSON file actually exists
        json_file = self.animations_dir / f"{clean_gloss}.json"
        return json_file.exists()

    def get_animation_info(self, gloss: str) -> Optional[Dict[str, Any]]:
        """Resolves animation details for an approved gloss. Returns None if unapproved or unavailable."""
        clean_gloss = gloss.strip().upper().replace("-", "_")
        if not self.is_approved(clean_gloss):
            return None

        entry = self._manifest[clean_gloss]
        json_file = self.animations_dir / f"{clean_gloss}.json"
        return {
            "gloss": clean_gloss,
            "status": entry["status"],
            "category": entry["category"],
            "url": f"/animations/slsl/{clean_gloss}.json",
            "file_path": str(json_file),
            "source_take": entry["take"],
        }

    def list_approved_glosses(self) -> List[str]:
        """Returns list of all approved gloss identifiers."""
        return sorted([g for g in self._manifest.keys() if self.is_approved(g)])

    def list_all_manifest_entries(self) -> List[Dict[str, Any]]:
        """Returns full list of all 383 manifest entries."""
        return list(self._manifest.values())


# Global singleton instance
_library_instance: Optional[AnimationLibrary] = None


def get_animation_library() -> AnimationLibrary:
    global _library_instance
    if _library_instance is None:
        _library_instance = AnimationLibrary()
    return _library_instance
