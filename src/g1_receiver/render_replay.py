"""Render recorded images only: never synthesize robot motion or a catch."""
import argparse
from pathlib import Path
import numpy as np


def main():
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensor', type=Path)
    parser.add_argument('--overview', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='Physics flight / static G1 reference / no catch controller')
    args = parser.parse_args()
    with np.load(args.overview, allow_pickle=False) as data:
        overview, overview_times = data['rgb'], data['timestamps_s']
    sensor, sensor_times = overview, overview_times
    if args.sensor:
        with np.load(args.sensor, allow_pickle=False) as data:
            sensor, sensor_times = data['rgb'], data['timestamps_s']
    times = overview_times
    if len(times) < 2 or len(sensor_times) != len(sensor) or not len(sensor) or (np.diff(sensor_times) <= 0).any():
        raise ValueError('Replays must have increasing timestamps matching their frames')
    if sensor_times[0] > times[0]:
        raise ValueError('Sensor replay must start by the first overview frame')
    # Display most recent captured frame, never a future image or synthesized motion.
    sensor_indices = np.searchsorted(sensor_times, times, side='right')-1
    deltas = np.diff(times)
    if (deltas <= 0).any() or not np.allclose(deltas, deltas[0], rtol=.01):
        raise ValueError('Video export requires uniform increasing timestamps')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=1/float(np.median(deltas)), codec='libx264', macro_block_size=1) as writer:
        for index, left in enumerate(overview):
            sensor_index = sensor_indices[index]
            right = sensor[sensor_index]
            frame = Image.new('RGB', (1280, 440), '#121a23')
            if args.sensor:
                frame.paste(Image.fromarray(left).resize((640, 360)), (0, 44))
                frame.paste(Image.fromarray(right).resize((640, 360)), (640, 44))
            else:
                frame.paste(Image.fromarray(left).resize((640, 360)), (320, 44))
            draw = ImageDraw.Draw(frame)
            draw.text((16, 14), 'G1 RECEIVER / rendered simulation replay', fill='white')
            view_label = f'RGB capture t={sensor_times[sensor_index]:.3f}s' if args.sensor else 'Recorded spectator view'
            draw.text((655, 14), f'{view_label} / playback t={times[index]:.3f}s', fill='white')
            draw.text((16, 416), args.label, fill='white')
            writer.append_data(np.asarray(frame))
            if index == min(10, len(times)-1):
                frame.save(args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
