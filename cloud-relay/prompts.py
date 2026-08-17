"""
LaViRA prompt templates for the Unitree Go1 VLN task.

Three prompts are exported:
  get_todo_generator_prompt      – one-shot checklist generation
  get_navigation_prompt_text     – strategic LA navigation call
  get_tactical_eyes_prompt       – tactical VA bbox / NAVIGATE / STOP call
"""


def get_todo_generator_prompt():
    return """
        Your task is to create a dynamic checklist (TODO list) to complete the instruction based on the visual context.

        Requirements:
        - Break down the instruction into logical, sequential steps.
        - Use the visual information to identify landmarks or initial direction if possible.
        - Format as a Markdown checklist:
        - [ ] Step 1 description
        - [ ] Step 2 description

        Response format:
        Return ONLY the markdown checklist string. Do not use JSON.
        """


def get_navigation_prompt_text(
    instruction, global_target, current_todo_list, history_info, current_step=1, selectable_views=None,
):
    selectable_views = selectable_views or []
    selection_contract = ""
    if selectable_views:
        selection_contract = f"""
            **OBSERVATION SELECTION (MANDATORY)**:
            You receive one independent panorama image for every label below. Inspect EVERY labelled observation
            before deciding. The only legal selected_view values are: {", ".join(selectable_views)}.
            Copy one label exactly. Never output a turn angle or a rotation command: robot software derives the
            physical turn solely from selected_view.
        """
    return f"""
            **ROLE**: You are an intelligent humanoid robot navigator using a generic checklist to guide your actions.
            **MISSION**: "{instruction}"

            **Current TODO List**:
            {current_todo_list}

            **INPUT**: The robot samples a physical panorama at the labelled poses. Each observation label is both
            visible in its image and a possible future body heading.
            {selection_contract}

            **DECISION PROCESS**:
            1. Preserve completed checklist items, update the full TODO list, and work on the first incomplete item.
            2. Ground every object, landmark, and relation in the labelled observations. Do not invent unseen facts.
            3. For any relational goal (distance, order, side, nearest, farthest, before, after, or similar):
               - identify the visible reference landmark or object;
               - identify all visible candidate targets relevant to the current TODO item;
               - compare candidates using the labelled observations before choosing one.
            4. Choose the observation that has the strongest visual evidence for progress on the current TODO item.
               Do not choose an object merely because it is most central or visually salient when the task requires
               a relation to another landmark.
            5. If evidence is insufficient, keep the TODO item incomplete and choose the observation most likely to
               resolve the ambiguity safely.
            6. Mark an item [x] only when the current observations and execution history support arrival.
            7. **Stop Decision**: Set "stop": true if you are sure you have reached the final goal.

            **HISTORY ANALYSIS**:
            {history_info}

            **JSON RESPONSE FORMAT**:
            {{
                "progress_analysis": "One short sentence summarizing current progress (MAX 30 words)",
                "reasoning": "Brief evidence citing the relevant observation labels and, when applicable, the reference and candidates compared",
                "updated_todo_list": "The full updated Markdown checklist string (with [x] and [ ])",
                "stop": true or false,
                "selected_view": "One exact offered observation label" (ONLY if stop is false),
                "expected_landmark": "What to look for next" (ONLY if stop is false)
            }}

            **CRITICAL**:
            - If stop is false, selected_view is required and must be an exact offered label.
            - If stop is true, omit selected_view.
            - progress_analysis and reasoning MUST be concise and evidence-based. Do NOT write long image descriptions.
            - Output ONLY the JSON object.
            - Do NOT output markdown code blocks (```json ... ```).
            - Do NOT output any explanatory text outside the JSON.
            """


def get_tactical_eyes_prompt(instruction, global_target, strategic_goal, strategic_stop, progress_analysis=""):
    return f"""
**ROLE**: You are a humanoid robot navigator's TACTICAL EYES.
**MISSION**: "{instruction}"
**GLOBAL TARGET**: "{global_target}"
**PROGRESS ANALYSIS**: "{progress_analysis}"
**CURRENT STRATEGY**: "{strategic_goal}"
**STRATEGIC STOP SIGNAL**: {strategic_stop}

**INPUT**: You are looking at the CURRENT VIEW after turning.

**TASK**:
1. **Verification**: Do you see the object/area mentioned in "CURRENT STRATEGY"?
2. **Targeting**: Draw a Bounding Box (bbox_2d) around the best navigation target to move forward.
   - If the GLOBAL TARGET is visible, box it.
   - If not, box the landmark mentioned in CURRENT STRATEGY.
3. **Action Decision (NAVIGATE vs STOP)**:
   - **NAVIGATE**: If the target is far away or not centered.
   - **STOP**: ONLY if the GLOBAL TARGET is clearly visible, centered, and occupies more than 20% of the image height.
   - **SPECIAL CASE**: If STRATEGIC STOP SIGNAL is True, verify if we are indeed at the goal. If yes, output STOP.

**JSON FORMAT**:
{{
    "visual_check": "I see [Object]...",
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "Name of the object",
    "stop_reasoning": "Reason if stopping"
}}
"""
