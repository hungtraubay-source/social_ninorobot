# Output of one chronological VLM scene. `header` is the fourth/newest RGB
# frame, in the same map-time context as the corresponding People message.
std_msgs/Header header

# Runtime instant when Qwen completed this request, for latency/staleness
# diagnostics. It is not substituted for the observation timestamp above.
builtin_interfaces/Time inference_stamp

# Parsed results only for IDs visible in the newest frame. The original model
# text is retained so malformed or unexpected model output can be debugged.
social_perception/VlmPersonState[] states
string raw_response
