# Webcam Classifications for V-JEPA2

A background service that continuously captures videos from your webcam and sends them to your V-JEPA2 service for classification.

## Setup

1. **Install dependencies** (if not already available):
   ```bash
   pip install opencv-python requests PyYAML numpy
   ```

2. **Start your V-JEPA2 service** (in another terminal):
   ```bash
   cd cloud-run/service
   python main.py
   ```

3. **Configure the webcam monitor** by editing `webcam/config.yaml` if needed (optional - defaults should work).

## Usage

### Start the Monitor
```bash
./webcam/start.sh
```

### Stop the Monitor
```bash
./webcam/stop.sh
```

### Check Status
```bash
./webcam/status.sh
```

### View Live Logs
```bash
tail -f webcam/webcam_clf.log
```

## How It Works

1. **Video Capture**: Captures 3-second videos every 2 seconds (creating 1-second overlap)
2. **Classification**: Sends each video to the V-JEPA2 service at `http://localhost:8080/classify-upload`
3. **Logging**: Records all activity including classification results with timestamps

## Configuration

Edit `webcam/config.yaml` to customize:

- **Service URL**: Change the V-JEPA2 service endpoint
- **Camera**: Specify camera index if auto-detection fails
- **Video Settings**: Adjust duration, overlap, resolution
- **Logging**: Change log level and format

## Troubleshooting

1. **Camera not found**: Check `webcam/config.yaml` and set specific camera index
2. **Service connection failed**: Ensure V-JEPA2 service is running at configured URL
3. **Permission denied**: Run `chmod +x *.sh` to make scripts executable
4. **Dependencies missing**: Run `pip install -r requirements.txt`
