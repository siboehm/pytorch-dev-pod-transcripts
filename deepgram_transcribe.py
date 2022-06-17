from asyncio import base_tasks
from xmlrpc.client import TRANSPORT_ERROR
from deepgram import Deepgram
from pathlib import Path
import asyncio, json
import os

DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

AUDIO_SRC_DIR = "data/audio_src/"
RAW_TRANSCRIPT_DIR = "data/raw_transcripts/"
MIMETYPE = "audio/mpegaudio/mpeg"  # mp3


async def get_transcript(filepath, dg_client):
    with open(filepath, "rb") as audio:
        transcript_path = Path(RAW_TRANSCRIPT_DIR) / (filepath.stem + ".json")

        if transcript_path.exists():
            print("Transcript already exists: " + str(transcript_path))
        else:
            print("Processing " + str(transcript_path))

            source = {"buffer": audio, "mimetype": MIMETYPE}
            options = {"punctuate": True, "language": "en", "model": "general-enhanced"}

            response = await dg_client.transcription.prerecorded(source, options)
            with open(transcript_path, "w") as f:
                json.dump(response, f, indent=4)
            print("Done " + str(transcript_path))


async def transcribe_audio():
    # Initializes the Deepgram SDK
    dg_client = Deepgram(DEEPGRAM_API_KEY)

    file_paths = list(Path(AUDIO_SRC_DIR).glob("**/*"))
    batch_size = 3
    for i in range(0, len(file_paths), batch_size):
        async_tasks = []
        for file_path in file_paths[i : i + batch_size]:
            async_tasks.append(get_transcript(file_path, dg_client))
        await asyncio.gather(*async_tasks)


def get_title_and_date(file_name):
    print(file_name)
    if len(file_name.split("-")[0]) == 4:
        date = file_name.split("-")[0:3]
        assert len(date[2]) == 2
    else:
        date_str = file_name.split("-")[0]
        date = [date_str[0:4], date_str[4:6], date_str[6:8]]
    title = " ".join(
        [w[0].upper() + w[1:] for w in file_name.split("-") if w and not w.isdigit()]
    )
    return title, date

def generate_markdown():
    episode_list = []
    for file_path in Path(RAW_TRANSCRIPT_DIR).glob("**/*"):
        with open(file_path) as f:
            title, date = get_title_and_date(file_path.stem)
            episode_list.append((title, date, file_path))
            print("Generating", title, date)
            header = f"""---
layout: post
title: "{title}"
date: {date[0]}-{date[1]}-{date[2]}
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# {title}

"""
            data = json.load(f)
            # save as markdown
            with open(
                Path("episodes")
                / ("-".join(date) + "-" + title.replace(" ", "-") + ".md"),
                "w",
            ) as f:
                f.write(header)
                for line in data["results"]["channels"][0]["alternatives"][0][
                    "transcript"
                ].split("."):
                    f.write(line.strip() + ".\n")
    with open("episodes.md", "w") as f:
        f.write("\n## Episodes\n\n")
        for title, date, file_path in episode_list:
            f.write(f"* {'-'.join(date)} [{title}]({file_path})\n")

if __name__ == "__main__":
    asyncio.run(transcribe_audio())
    generate_markdown()

