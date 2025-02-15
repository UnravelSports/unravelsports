from IPython.display import HTML

from typing import List, Union


def show(
    video_path: Union[List[str], str],
    width=640,
    height=480,
    as_ipython_display: bool = True,
):
    if isinstance(video_path, str):
        video_path = [video_path]

    html_videos = "".join(
        [
            f'<video id="video{i}" src="{x}" width="{width}" height="{height}"></video>'
            for i, x in enumerate(video_path)
        ]
    )
    html_play_btns = "".join(
        [
            f'document.getElementById("video{i}").play();'
            for i, _ in enumerate(video_path)
        ]
    )
    html_pause_btns = "".join(
        [
            f'document.getElementById("video{i}").pause();'
            for i, _ in enumerate(video_path)
        ]
    )

    html_string = f"""
    <div style="display: flex; justify-content: space-around; align-items: center;">
        {html_videos}
    </div>
    <br>
    <div style="text-align: center;">
        <button onclick="playVideos()" style="background-color: #4CAF50; border: none; color: white; padding: 10px 20px; font-size: 16px; margin: 5px; border-radius: 5px; cursor: pointer;">
            <i class="fa fa-play"></i> Play
        </button>
        <button onclick="pauseVideos()" style="background-color: #f44336; border: none; color: white; padding: 10px 20px; font-size: 16px; margin: 5px; border-radius: 5px; cursor: pointer;">
            <i class="fa fa-pause"></i> Pause
        </button>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/js/all.min.js"></script>
    <script>
        function playVideos() {{
            {html_play_btns}
        }}
        function pauseVideos() {{
            {html_pause_btns}
        }}
    </script>
    """
    if as_ipython_display:
        return HTML(html_string)
    return html_string
