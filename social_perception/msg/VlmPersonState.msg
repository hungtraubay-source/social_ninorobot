# One semantic VLM decision for one currently visible ByteTrack identity.
# `track_id` is the integer rendered in all four VLM input frames, while
# `person_id` is the matching stable ROS identifier (normally "person_<id>").
int32 track_id
string person_id

# State emitted by the fine-tuned VLM, for example "crossing". Consumers must
# treat this as semantic metadata; pose, velocity, and safety geometry remain
# owned by the RGB-D/ByteTrack perception pipeline.
string state
