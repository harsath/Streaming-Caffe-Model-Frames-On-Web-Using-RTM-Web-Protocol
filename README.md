# Streaming Caffe's Processed Frames on Web With Internal IP
### ✅ What's RTM Protocol?
#### It's a Web based protocol for streaming Media application on to web using HTTP via TCP/IP. It splits frames into fragmented Bytes with JPEG Compression and Sends to the Web using the an IP on specific Port. The fragmented data is 64 bytes for audio data, and 128 bytes for video data. It supportes Type-9 Bandwidth on CM Structure.
![cool__web](https://user-images.githubusercontent.com/30565388/64517212-815e7e80-d30d-11e9-8381-a9c2c629300a.png)
### 🧵 Here are using Threads for Multiple clients requesting the IP
#### Simultaneous execution of two or more parts of a program to maximum utilize the CPU time. A multithreaded program contains two or more parts that can run concurrently for reducing serverside latency
### ❓How to run?
> $python3 streamer_web_MAIN.py --prototxt deploy.prototxt.txt --model mobilenet.caffemodel --ip 0.0.0.0 --port 86330
#### 📝 I post this Code to Help people who are looking to Stream their DL Model's Output into a Web Protocol by showing them a Simple example so do what ever you want with my code.
