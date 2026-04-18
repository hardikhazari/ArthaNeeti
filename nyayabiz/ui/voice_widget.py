# ─────────────────────────────────────────────────────────────────────────────
# VOICE QUERY WIDGET  (ipywidgets — for Databricks notebook UI)
# ─────────────────────────────────────────────────────────────────────────────

import os
import tempfile

import ipywidgets as widgets
from IPython.display import display, clear_output


def launch_voice_widget(run_rag_from_audio_fn, print_rag_result_fn):
    """Display an ipywidgets file-upload UI for voice queries.

    Args:
        run_rag_from_audio_fn: A callable(audio_path) -> dict that transcribes
            the audio and runs the RAG pipeline. Must be defined in the notebook
            (wraps Whisper + run_rag).
        print_rag_result_fn: A callable(dict) -> None for pretty-printing results.
    """

    # 1. Create a File Upload button
    audio_uploader = widgets.FileUpload(
        accept='audio/*',  # Accept all audio files (.wav, .mp3, .m4a)
        multiple=False,
        description='🎙️ Upload Audio',
        button_style='warning',  # Makes the button yellow/orange
        layout=widgets.Layout(width='200px')
    )

    # 2. Output area for the results
    output = widgets.Output()

    # 3. Define what happens when a file is uploaded
    def on_audio_upload(change):
        with output:
            clear_output()

            # Get the uploaded file data
            uploaded_file = audio_uploader.value

            if not uploaded_file:
                return

            # Extract filename and content (ipywidgets 8.x format)
            if isinstance(uploaded_file, tuple):
                file_info = uploaded_file[0]
                file_name = file_info['name']
                file_content = file_info['content']
            else:  # ipywidgets 7.x format
                file_name = list(uploaded_file.keys())[0]
                file_content = uploaded_file[file_name]['content']

            print(f"✅ Received: {file_name}")
            print("⏳ Processing through Whisper & NyayaBiz...")

            # Save the bytes to a temporary file so Whisper can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(file_content)
                temp_audio_path = temp_audio.name

            try:
                # ----> Call the pipeline from the notebook <----
                audio_out = run_rag_from_audio_fn(temp_audio_path)

                print("\n" + "=" * 50)
                print("⚖️ NYAYABIZ LEGAL ADVISOR RESPONSE")
                print("=" * 50)
                print(f"🗣️ Transcribed Query ({audio_out['whisper_language']}): {audio_out.get('question', '...')}\n")

                # Print your result
                print_rag_result_fn(audio_out)

            except Exception as e:
                print(f"\n❌ Error processing audio: {e}")

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

            # Reset the uploader for the next question
            audio_uploader.value = tuple()  # or {} for older ipywidgets

    # 4. Bind the function to the uploader
    audio_uploader.observe(on_audio_upload, names='value')

    # 5. Display the UI
    display(widgets.VBox([
        widgets.HTML("<h3>🗣️ NyayaBiz Voice Query</h3><p>Upload a voice memo to get legal analysis.</p>"),
        audio_uploader,
        output
    ]))
