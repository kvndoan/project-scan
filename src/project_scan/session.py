from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .slam import SlamBackend, SlamUpdate


class ScanSession:
    def __init__(
        self,
        name: str,
        backend: SlamBackend,
        zones_dir: str | Path = "zones",
    ) -> None:
        name = name.strip()

        if not name:
            raise ValueError(
                "Scan name cannot be empty"
            )

        self.name = name
        self._backend = backend
        self._zones_dir = Path(zones_dir)

        self._started_at: datetime | None = None
        self._zone_dir: Path | None = None
        self._updates: list[SlamUpdate] = []

    @property
    def active(self) -> bool:
        return self._started_at is not None

    @property
    def zone_dir(self) -> Path | None:
        return self._zone_dir

    def start(self) -> None:
        if self.active:
            raise RuntimeError(
                "Scan session is already active"
            )

        started_at = datetime.now(
            timezone.utc
        )

        zone_id = started_at.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        zone_dir = (
            self._zones_dir
            / zone_id
        )

        zone_dir.mkdir(
            parents=True
        )

        self._backend.start(
            recording_path=(
                zone_dir / "recording"
            )
        )

        self._started_at = started_at
        self._zone_dir = zone_dir
        self._updates = []

    def wait_for_update(
        self,
    ) -> SlamUpdate:
        if not self.active:
            raise RuntimeError(
                "Scan session is not active"
            )

        update = (
            self._backend.wait_for_update()
        )

        self._updates.append(update)

        return update

    def stop(self) -> Path:
        if (
            not self.active
            or self._zone_dir is None
            or self._started_at is None
        ):
            raise RuntimeError(
                "Scan session is not active"
            )

        stopped_at = datetime.now(
            timezone.utc
        )

        started_at = self._started_at
        zone_dir = self._zone_dir

        self._backend.stop()

        self._save_trajectory(
            zone_dir
        )

        self._backend.export_map(
            zone_dir / "map.ply"
        )

        self._save_metadata(
            zone_dir,
            started_at,
            stopped_at,
        )

        self._started_at = None
        self._zone_dir = None
        self._updates = []

        return zone_dir

    def _save_trajectory(
        self,
        zone_dir: Path,
    ) -> None:
        timestamps = np.array(
            [
                update.timestamp
                for update
                in self._updates
            ],
            dtype=np.float64,
        )

        tracking_status = np.array(
            [
                update.tracking_status
                for update
                in self._updates
            ]
        )

        if self._updates:
            camera_to_world = np.stack(
                [
                    update.camera_to_world
                    for update
                    in self._updates
                ]
            )
        else:
            camera_to_world = np.empty(
                (0, 4, 4),
                dtype=np.float64,
            )

        np.savez_compressed(
            zone_dir / "trajectory.npz",
            timestamp=timestamps,
            tracking_status=tracking_status,
            camera_to_world=camera_to_world,
        )

    def _save_metadata(
        self,
        zone_dir: Path,
        started_at: datetime,
        stopped_at: datetime,
    ) -> None:
        final_update = (
            self._updates[-1]
            if self._updates
            else None
        )

        metadata = {
            "id": zone_dir.name,
            "name": self.name,
            "started_at": (
                started_at.isoformat()
            ),
            "stopped_at": (
                stopped_at.isoformat()
            ),
            "duration_seconds": (
                stopped_at
                - started_at
            ).total_seconds(),
            "trajectory_samples": len(
                self._updates
            ),
            "final_tracking_status": (
                final_update.tracking_status
                if final_update
                else None
            ),
            "mapping": (
                asdict(
                    final_update.mapping
                )
                if final_update
                else None
            ),
            "files": {
                "trajectory": (
                    "trajectory.npz"
                ),
                "map": "map.ply",
                "recording": "recording",
            },
        }

        with (
            zone_dir
            / "metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

            file.write("\n")