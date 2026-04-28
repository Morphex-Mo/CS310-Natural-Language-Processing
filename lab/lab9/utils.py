# Original code from https://github.com/haizelabs/j1-micro/tree/master
# Adopted and modified by Yang Xu (xuyang@sustech.edu.cn), 2026-04-27

import textwrap
import re
from typing import List


def judge_system_prompt() -> str:
    return textwrap.dedent(
        """
    You are an expert XML wrangler. You must respond in the following format, regardless of the input:
    
    <specific_criteria>
    ...
    </specific_criteria>
    <analysis>
    ...
    </analysis>
    <scores>
    \\boxed{{..., ...}}
    </scores>

    Please only respond in English.
    """
    ).strip()


judge_prompt_template = textwrap.dedent(
    """
    You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
    Given the context of the conversation (the last round is the User's query) and multiple responses from the Assistant, you need to refer to the [General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score upon them.
    Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.
    Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

    #### Evaluation Criteria ####
    1. Instruction Adherence:
    - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.
    - Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.
    - Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.
    - Not Adhered (1-2 points): The response does not meet any instructions.
    Example: If the question requires three examples and the response provides only one, it falls under "Partially Adhered."
    2. Usefulness:
    - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.
    - Useful but Incomplete (6-8 points): The response provides some useful information, but lacks details or accuracy.
    - Limited Usefulness (3-5 points): The response offers little useful information, with most content being irrelevant or incorrect.
    - Useless or Incorrect (1-2 points): The response is completely irrelevant or incorrect.
    Example: If there are factual errors in the response but the overall direction is correct, it falls under "Useful but Incomplete."
    3. Level of Detail:
    - Very Detailed (9-10 points): The response includes ample details covering all aspects of the issue.
    - Detailed but Slightly Lacking (6-8 points): The response is fairly detailed but misses some important details.
    - Basically Detailed (3-5 points): The response provides some details but is not thorough enough overall.
    - Not Detailed (1-2 points): The response is very brief and lacks necessary details.
    Example: If the response provides only a simple conclusion without an explanation, it falls under "Not Detailed."
    4. Relevance:
    - Highly Relevant (9-10 points): The response is highly relevant to the question, with information closely aligned with the topic.
    - Generally Relevant (6-8 points): The response is generally relevant but includes some unnecessary information.
    - Partially Relevant (3-5 points): The response has a lot of content that deviates from the topic.
    - Not Relevant (1-2 points): The response is completely irrelevant.
    Example: If the response strays from the topic but still provides some relevant information, it falls under "Partially Relevant."

    #### Conversation Context ####
    {conversation_context_query}
    #### Responses to be Scored ####
    [The Begin of Response A]
    {response_a}
    [The End of Response A]
    [The Begin of Response B]
    {response_b}
    [The End of Response B]
    #### Output Format Requirements ####

    Output with three lines
    <specific_criteria>
    [Other potential criteria specific to the query and the context, and the weights of each criteria.]
    </specific_criteria>
    <analysis>
    [Compare different responses based on given Criteria.]
    </analysis>
    <scores>
    [The overall comprehensive score of all responses in order, separate by comma in the boxed, e.g., \\boxed{{x, x}} if there exists 2 responses.]
    </scores>
    """
).strip()


def judge_prompt_format(
    conversation_context_query: str, response_a: str, response_b: str
) -> str:
    """
    See page 40 of https://arxiv.org/abs/2504.02495
    """

    return judge_prompt_template.format(
        conversation_context_query=conversation_context_query,
        response_a=response_a,
        response_b=response_b,
    )


class FormatError(Exception):
    pass


def extract_scores(raw_response: str) -> List[float]:
    """
    Extract Judge scores from the raw response.
    Expects the following format:
        ---
        <specific_criteria>...</specific_criteria>
        <analysis>...</analysis>
        <scores>\boxed{x, y}</scores>
        ---
    """
    match = re.search(r"<scores>(.*?)</scores>", raw_response, re.DOTALL)
    if not match:
        raise FormatError("No Judge scores found in response")

    boxed_match = re.search(r"\\boxed{([\d.]+),\s*([\d.]+)}", match.group(1))
    if not boxed_match:
        raise FormatError("No boxed scores found in scores tag")

    try:
        return [float(boxed_match.group(1)), float(boxed_match.group(2))]
    except ValueError:
        raise FormatError("Invalid score format in boxed response")