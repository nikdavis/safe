## Purpose

This software is was intended to be a safety device for automobiles and motorcycles.
It tracks vehicular movement behind a vehicle, and alerts with sound on an impending
collision.

## Development and running

### Mac / OS X
You will need [brew](https://brew.sh/) in order to install opencv.

Install openCV 2:
```
brew install opencv@2 --with-ffmpeg -v
brew link opencv@2 --force
```

Compile the code:
`make clean && make` or `make clean && make debug` (the latter can be used with GDB)

Run the binary ...

With a video file:
`./safe -v <video_file>`

or with your webcam:
`./safe -c <index_of_your_webcam>` (usually 0 unless you have multiple webcams)
