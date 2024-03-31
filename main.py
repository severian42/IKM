import json
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from llm_handler import send_to_chatgpt
from params import OUTPUT_FILE_PATH, NUM_WORKERS

from system_messages import (
    SYSTEM_MESSAGES_ORCA,
)
from topics import TOPICS

def save_to_file(data, file_path):
    print(f"Writing data to {file_path}")  # Debug print
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')

code_data = False

SYSTEM_MESSAGES = SYSTEM_MESSAGES_ORCA
PROMPT_1 = """For the following SUBJECT_AREA, create a narrative that explores a character's inner thoughts, emotions, and experiences as they navigate a challenging situation or dilemma. The story should delve into a narrow but significant aspect of the SUBJECT_AREA, revealing deep insights and provocative ideas. Each narrative must be complex, multi-layered, and structured to unveil the true nature of the subject area through the character's perspective. Your dataset should be presented as an interconnected network of story/inner dialogue clusters, each documented and enhanced with metadata, internal thought processes, literary techniques, and thematic implications.

### Structured Guidelines:

1. **Core Interactions**: 
   - **Story**: Craft a compelling narrative that immerses the reader in the character's world, ensuring it's emotionally engaging, thought-provoking, and relevant to the **SUBJECT_AREA**.
   - **Inner Dialogue**: Reveal the character's inner thoughts, feelings, and mental processes as they grapple with the central dilemma or situation, providing a rich, introspective exploration of the subject matter.

2. **Supportive Nodes**:
   - **Metadata**: Include attributes like genre, tone, point of view, and target audience for each story/inner dialogue pair.
   - **Internal Thought Processes and Mapping**: Document the creative process behind developing the narrative and the character's inner dialogue, including the inspiration, themes, and intended impact.
   - **Contextual Backgrounds**: Provide historical, cultural, or situational contexts that enrich the understanding of the character's experiences and the story's setting.
   - **Literary Techniques**: Detail the literary devices, narrative structures, and stylistic choices employed in crafting the story and inner dialogue.
   - **Related Concepts and Themes**: Highlight the key ideas, motifs, and philosophical questions explored through the narrative and the character's introspection.
   - **Thematic Implications**: Discuss the broader implications and universal truths that the story and inner dialogue illuminate about the human condition and the **SUBJECT_AREA**.

3. **Documentation and Edges**:
   - Use markdown to document each component, ensuring clarity and coherence across the dataset. Include tags, links, and navigation paths to facilitate integration and exploration within the story network.

Your goal is to leverage this structured approach to create a dataset that serves as a comprehensive story network, facilitating the exploration and application of **SUBJECT_AREA** through the power of narrative and introspection. Use markdown for clear documentation, and ensure the dataset is immersive, thought-provoking, and emotionally resonant.
"""

msg_context = {"role": "system", "content": str(PROMPT_1)}


def generate_data(
    topic_selected,
    system_message_generation,
    system_message_selected,
    OUTPUT_FILE_PATH,
):
    system_contexts = [
        system_message_generation,
        system_message_selected,
    ]

    user_prompts = [f"SUBJECT_AREA: {topic_selected}"]
    gpt_outputs = []

    for pp in range(len(system_contexts)):
        msg_list = []
        msg_system = {"role": "system", "content": str(system_contexts[pp])}
        msg_list.append(msg_system)
        msg_prompt = {"role": "user", "content": user_prompts[pp]}
        msg_list.append(msg_prompt)

        msg_list = [msg for msg in msg_list if 'content' in msg and msg['content'].strip()]

        chatgpt_response, gpt_usage = send_to_chatgpt(msg_list)
        gpt_generated_prompt = chatgpt_response



        user_prompts.append(str(gpt_generated_prompt))
        gpt_outputs.append(gpt_generated_prompt)

    data = {
        "system": system_contexts[1],
        "instruction": str(user_prompts[1]),
        "response": str(gpt_outputs[1]),
    }

    #with open(OUTPUT_FILE_PATH, "a") as output_file:
    #    output_file.write(json.dumps(data) + "\n")


        # Call save_to_file right here
    save_to_file(data, OUTPUT_FILE_PATH)

    return data, gpt_usage


def main():
    nn = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Create a list of futures, one for each topic
        futures = []
        for _ in range(NUM_WORKERS):
            topic_number = np.random.randint(0, len(TOPICS))
            topic_selected = TOPICS[topic_number]
            system_message_number = np.random.randint(0, len(SYSTEM_MESSAGES))
            system_message_selected = SYSTEM_MESSAGES[system_message_number]
            system_message_generation = PROMPT_1
            futures.append(
                executor.submit(
                    generate_data,
                    topic_selected,
                    system_message_generation,
                    system_message_selected,
                    OUTPUT_FILE_PATH,
                )
            )

        # Wait for all futures to complete
        for future in futures:
            data, gpt_usage = future.result()
            if gpt_usage is not None:
                nn += 1
                print(data)
                print(
                    f"GPT Generation {nn} Complete, Token usage: {gpt_usage}, Failed: {failed}"
                )
            else:
                failed += 1
            print("=" * 132)


while True:
    main()
