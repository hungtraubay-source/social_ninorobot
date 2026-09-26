"""Lossless JSON transport for the deploy-time ConstraintField."""

import json

from social_rl.constraint_field import ConstraintField, Zone, ZoneSample


def field_to_json(field: ConstraintField) -> str:
    """Serialize without the rounding used by human-facing to_dict()."""
    payload = {
        'schema': 1,
        'timestamp': field.timestamp,
        'frame': field.frame,
        'horizon_steps': field.horizon_steps,
        'dt': field.dt,
        'zones': [
            {
                'zone_id': zone.zone_id,
                'track_ids': list(zone.track_ids),
                'scene_type': zone.scene_type,
                'hardness': zone.hardness,
                'confidence': zone.confidence,
                'valid_from': zone.valid_from,
                'valid_to': zone.valid_to,
                'trajectory_of_zone': [
                    {
                        't': sample.t,
                        'shape': sample.shape,
                        'center': list(sample.center),
                        'size': list(sample.size),
                        'weight': sample.weight,
                        'orientation': sample.orientation,
                    }
                    for sample in zone.trajectory_of_zone
                ],
            }
            for zone in field.zones
        ],
    }
    return json.dumps(
        payload, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def field_from_json(encoded: str) -> ConstraintField:
    """Deserialize one field published by the deploy field node."""
    payload = json.loads(encoded)
    if payload.get('schema') != 1:
        raise ValueError('unsupported ConstraintField transport schema')
    zones = []
    for item in payload['zones']:
        samples = tuple(
            ZoneSample(
                t=float(sample['t']),
                shape=str(sample['shape']),
                center=tuple(float(value) for value in sample['center']),
                size=tuple(float(value) for value in sample['size']),
                weight=float(sample['weight']),
                orientation=float(sample['orientation']))
            for sample in item['trajectory_of_zone'])
        zones.append(Zone(
            zone_id=str(item['zone_id']),
            track_ids=tuple(item['track_ids']),
            scene_type=str(item['scene_type']),
            hardness=str(item['hardness']),
            confidence=float(item['confidence']),
            valid_from=float(item['valid_from']),
            valid_to=float(item['valid_to']),
            trajectory_of_zone=samples))
    return ConstraintField(
        timestamp=float(payload['timestamp']),
        frame=str(payload['frame']),
        horizon_steps=int(payload['horizon_steps']),
        dt=float(payload['dt']),
        zones=tuple(zones))
