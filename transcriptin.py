import argparse
import datetime
import logging
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import openai
from llama_index.response.schema import RESPONSE_TYPE
from pydub import AudioSegment

from constants import DEFAULT_TARGET_FILE_SIZE, FOLDERPATH_DOCUMENTS
from construct_index import load_variables, query_with_index, save_variables

# logging.disable()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

TRANSCRIPT_PROMPT = """
こんにちは。今日は、いいお天気ですね。
"""

# TRANSCRIPT_PROMPT = """
# Hello, welcome to my lecture.
# """

current_time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def show_response(response: RESPONSE_TYPE, user_input: str):
    """
    Output responses in stream format.
    """

    print("==========")
    print("Query:")
    print(user_input)
    print("Answer:")
    print(response)
    print("==========\n")

    # node_list = response.source_nodes

    # for node in node_list:
    #     node_dict = node.dict()
    #     print("----------")

    #     print("Node ID:")
    #     print(f"{node_dict['node']['id_']}")

    #     print("Cosine Similarity:")
    #     print(f"{node_dict['score']}\n")

    #     print("Reference text:")
    #     print(f"{node_dict['node']['text']}")
    #     print("----------\n")


# Extract the audio source, and write it into a file.
def extract_audio_from_media(original_file_path):
    """
    Extract audio from video file, adjusting the size of the audio file to meet the 25MB limit of OpenAI's Whisper API.
    ref: https://dev.classmethod.jp/articles/openai-api-whisper-about-data-limit/
    """
    logging.debug("original_file_path: %s", original_file_path)

    with tempfile.NamedTemporaryFile(suffix=original_file_path.suffix) as audio_file:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(original_file_path),
                "-codec:a",
                "copy",
                "-vn",
                audio_file.name,
            ],
            check=False,
        )

        audio_file_size = pathlib.Path(audio_file.name).stat().st_size

        logging.debug("audio_file_size: %s", audio_file_size)

        if audio_file_size > DEFAULT_TARGET_FILE_SIZE:
            logging.info("This file need to be converted.")

            audio_segment = AudioSegment.from_file(str(audio_file.name))
            audio_length_sec = len(audio_segment) / 1000
            target_kbps = int(
                math.floor(
                    DEFAULT_TARGET_FILE_SIZE * 8 / audio_length_sec / 1000 * 0.95
                )
            )

            logging.debug("target_kbps: %s", target_kbps)

            if target_kbps < 8:
                assert f"{target_kbps=} is not supported"

            with tempfile.NamedTemporaryFile(suffix=".mp4") as converted_file:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(audio_file.name),
                        "-codec:a",
                        "aac",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-b:a",
                        f"{target_kbps}k",
                        converted_file.name,
                    ],
                    check=False,
                )
                converted_file_size = pathlib.Path(converted_file.name).stat().st_size

                logging.debug("converted_file_size: %s", converted_file_size)

                with open(converted_file.name, "rb") as file:
                    generate_transcript(file)
        else:
            with open(audio_file.name, "rb") as file:
                generate_transcript(file)


def generate_transcript(audio_file):
    """
    Transcribe the text from the audio_file and output it under FOLDERPATH_DOCUMENTS.
    """
    logging.debug("audio_file: %s", audio_file)

    os.makedirs(FOLDERPATH_DOCUMENTS, exist_ok=True)
    output_file = pathlib.Path(
        f"./{FOLDERPATH_DOCUMENTS}/transcription_{current_time_str}.txt"
    )

    transcript = openai.Audio.transcribe(
        file=audio_file,
        model="whisper-1",
        response_format="json",
        # Input language.
        # ref: https://platform.openai.com/docs/guides/speech-to-text/supported-languages
        language="ja",
        # Prompting
        # ref: https://platform.openai.com/docs/guides/speech-to-text/prompting
        prompt=TRANSCRIPT_PROMPT,
    )

    transcript_result = transcript["text"]

    with open(output_file, "wt", encoding="utf-8") as file:
        file.write(str(transcript_result))


def main():
    """
    Extract audio from video file and transcribe.
        $ python3 ./transcriptin.py -f ./sample.mp4
    Execute query.
        $ python3 ./transcriptin.py
    """
    parser = argparse.ArgumentParser(
        description="Extract audio data from movie, or update indeces with transcripted data."
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "-f",
        "--file",
        help="original movie file path.",
    )

    args = parser.parse_args()

    if args.file:
        original_file_path = pathlib.Path(args.file)
        extract_audio_from_media(original_file_path)
        sys.exit()

    while True:
        user_input = input("Input query:")
        if user_input == "exit":
            # save_variables()
            break

        # load the variables at app startup
        load_variables()
        response = query_with_index(user_input)
        show_response(response, user_input)

        output_file = pathlib.Path(f"./data/summary_{current_time_str}.txt")
        with open(output_file, "wt", encoding="utf-8") as file:
            file.write(str(response))


if __name__ == "__main__":
    main()
