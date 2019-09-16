from pathlib import Path

from diff_processor import *

if __name__ == '__main__':

    video_records = [
        {
            'path': str(p),
            'frames': -1,
            'fps': {}
        }
        for p in [v for v in sorted(Path('video').iterdir()) if v.suffix == '.mp4']
    ]

    diff_processors = {
        'pixel': PixelDiff(thresh=0.2),
        'area': AreaDiff(thresh=0.2),
        'edge': EdgeDiff(thresh=0.2),
        'corner': CornerDiff(thresh=0.2),
        'hist': HistDiff(thresh=0.2),
        'hog': HOGDiff(thresh=0.2),
        'sift': SIFTDiff(thresh=0.2),
        'surf': SURFDiff(thresh=0.2),
    }

    for video in video_records:
        with VideoProcessor(video['path']) as vp:
            for frame in vp:
                pass
            video['frames'] = vp.index
        for dp_name in diff_processors:
            video['fps'][dp_name] = []

    for video in video_records:
        print(video['path'])
        path, frames = video['path'], video['frames']
        for dp_name, dp in diff_processors.items():
            time_start = time.time()
            dp.process_video(path)
            time_end = time.time()
            runtime = time_end - time_start
            video['fps'][dp_name].append(frames / runtime)
            print(f'{dp_name}: {frames / runtime : .2f}')

    for video in video_records:
        print(video['path'])
        for dp_name in diff_processors:
            fpses = video['fps'][dp_name]
            avg_fps = sum(fpses) / len(fpses)
            print(f'{dp_name}: {avg_fps : .2f}')
