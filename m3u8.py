import m3u8_To_MP4

if __name__ == '__main__':
    # 1. Download videos from uri.
    m3u8_To_MP4.multithread_download('http://videoserver.com/playlist.m3u8')

    # 2. Download videos from existing m3u8 files.
    m3u8_To_MP4.multithread_file_download('http://videoserver.com/playlist.m3u8',m3u8_file_path)

    # For compatibility, i reserve this api, but i do not recommend to you again.
    # m3u8_To_MP4.download('http://videoserver.com/playlist.m3u8')