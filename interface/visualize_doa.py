import json
import numpy as np
import cv2 as cv
import ffmpeg

def parse_ratio(s):
    a, b = s.split('/')
    return int(a)/int(b)

def main():
    video_input_file = "../audio/lumens2.mp4"
    audio_input_file = "../audio/lumens2_aligned_+36.wav"
    doa_file = "../audio/msgout.json"
    out_file = "output.mkv"
    camera_calibration_file = "calibration.npz"

    scan_points = None
    doas_potential = []
    doas_tracked = []

    # Load line by line
    with open(doa_file) as fin:
        for line in fin:
            msg = json.loads(line)
            if msg['type'] == "doas":

                data = np.zeros((len(msg['dirs']), 5), dtype=np.float32)
                
                for src_index in range(len(msg['dirs'])):
                    data[src_index, 0] = msg['dirs'][src_index]['type']
                    data[src_index, 1] = msg['dirs'][src_index]['x']
                    data[src_index, 2] = msg['dirs'][src_index]['y']
                    data[src_index, 3] = msg['dirs'][src_index]['z']
                    data[src_index, 4] = msg['dirs'][src_index]['energy']

                if msg['label'] == 'tracked':
                    doas_tracked.append(data)
                if msg['label'] == 'potential':
                    doas_potential.append(data)
            if msg['type'] == 'points':
                data = np.zeros((len(msg['points']), 3), dtype=np.float32)
                
                for point_index, point in enumerate(msg['points']):
                    data[point_index, 0] = point['x']
                    data[point_index, 1] = point['y']
                    data[point_index, 2] = point['z']

                scan_points = data

    doas_tracked = np.stack(doas_tracked, axis=0)
    doas_potential = np.stack(doas_potential, axis=0)

    doas = doas_tracked

    calibration = np.load(camera_calibration_file)
    mtx = calibration['mtx']
    dist = calibration['dist']
    rvec = np.array([0, 0, 0], dtype=np.float32)
    tvec = np.array([0, 0, 0], dtype=np.float32)

    scan_points_projected, _ = cv.projectPoints(scan_points, rvec, tvec, mtx, dist)


    probe = ffmpeg.probe(video_input_file)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    video_fps = parse_ratio(video_stream['avg_frame_rate'])
    audio_fps = 48000/128

    # Read timestamp for each frame?
    # result = ffmpeg.input(video_input_file).filter('showinfo').output('pipe:', format='null').run()
    # print(result)

    video_input_process = (
        ffmpeg
        .input(video_input_file)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    out_audio = (
        ffmpeg
        .input(audio_input_file)
        .filter("pan", "1c|c0<c7") # get channel 8 from the input file
    )

    out_video = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), framerate=video_fps)
    )

    video_output_process = (
        ffmpeg.output(out_video, out_audio, out_file, pix_fmt='yuv420p', vcodec='mjpeg', acodec='aac', **{'q:v': 3})
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    i = 0
    while True:
        in_bytes = video_input_process.stdout.read(width * height * 3)
        if not in_bytes:
            break

        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3]).copy()

        time = i / video_fps
        audio_frame = int(audio_fps * time)
        if audio_frame >= doas.shape[0]:
            break

        # Draw all scanned points
        for point in scan_points_projected:
            cv.circle(frame, point.flatten().astype(int), 2, (255, 0, 0), -1)

        # Draw doa points
        doa_frame = doas[audio_frame]
        for source_idx in range(doas.shape[1]):
            doa_type = doa_frame[source_idx, 0]
            doa_point = doa_frame[source_idx, 1:4]
            doa_energy = doa_frame[source_idx, 4]
            if doa_type != 0:
                doa_point_projected, _ = cv.projectPoints(doa_point, rvec, tvec, mtx, dist)
                cv.circle(frame, doa_point_projected.flatten().astype(int), int(doa_energy * 60), (0, 0, 255), -1)


        video_output_process.stdin.write(frame.astype(np.uint8).tobytes())
        i += 1

    video_input_process.terminate()
    video_output_process.stdin.close()
    video_output_process.wait()

main()