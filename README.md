
```
ros2 topic pub --once /track_request vision_msgs/msg/Detection2D "{
  header: {
    stamp: {
      sec: 0,
      nanosec: 0
    },
    frame_id: 'camera'
  },
  bbox: {
    center: {
      position: {
        x: 320.0,
        y: 240.0
      },
      theta: 0.0
    },
    size_x: 100.0,
    size_y: 100.0
  },
  id: '',
  results: []
}"
```