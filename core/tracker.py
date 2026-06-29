
import supervision as sv
import numpy as np
import torch
import torchreid


class GlobalIDManager:
    def __init__(self, threshold=0.85):
        self.registry = {}
        self.threshold = threshold
        self.next_id = 1

    def get_or_create_id(self, feature_vector):
        best_id = None
        best_sim = -1.0

        for gid, stored_vec in self.registry.items():
            sim = torch.nn.functional.cosine_similarity(feature_vector, stored_vec, dim=0).item()
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        if best_sim >= self.threshold and best_id is not None:
            self.registry[best_id] = (self.registry[best_id] + feature_vector) / 2.0
            return best_id

        new_id = self.next_id
        self.registry[new_id] = feature_vector
        self.next_id += 1
        return new_id


class EventTracker:
    def __init__(self, camera_configs):
        self.camera_configs = camera_configs
        self.trackers = {cam["id"]: sv.ByteTrack() for cam in camera_configs}
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name="osnet_x1_0",
            device="cuda"
        )
        self.id_manager = GlobalIDManager(threshold=0.85)
        self.camera_to_global_map = {cam["id"]: {} for cam in camera_configs}

        self.zones = {}
        for cam in camera_configs:
            coords = cam.get("line_zone_coordinates") or cam.get("line_zone")
            if coords:
                start_point = sv.Point(x=coords[0][0], y=coords[0][1])
                end_point = sv.Point(x=coords[1][0], y=coords[1][1])
                self.zones[cam["id"]] = sv.LineZone(start=start_point, end=end_point)

    def update(self, cam_id, frame, detections):
        if len(detections) == 0:
            return detections

        tracked = self.trackers[cam_id].update_with_detections(detections=detections)

        global_ids = [None] * len(tracked)
        crops = []
        crop_indices = []

        for i in range(len(tracked)):
            bbox = tracked.xyxy[i]
            local_id = tracked.tracker_id[i]

            if local_id in self.camera_to_global_map[cam_id]:
                global_ids[i] = self.camera_to_global_map[cam_id][local_id]
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                global_ids[i] = local_id
                continue

            crops.append(crop)
            crop_indices.append(i)

        if crops:
            features_batch = self.extractor(crops)
            for j, feature in enumerate(features_batch):
                i = crop_indices[j]
                local_id = tracked.tracker_id[i]
                global_id = self.id_manager.get_or_create_id(feature)
                self.camera_to_global_map[cam_id][local_id] = global_id
                global_ids[i] = global_id

        tracked.tracker_id = np.array(global_ids)

        if cam_id in self.zones:
            zone = self.zones[cam_id]
            # zone.trigger(detections=tracked)

        return tracked

    def get_total_count(self):
        total_in = 0
        total_out = 0
        for zone in self.zones.values():
            total_in += zone.in_count
            total_out += zone.out_count
        return total_in + total_out

    def get_total_unique_ids(self):
        return len(self.id_manager.registry)
