import os
import sys
import math
from typing import Dict, List
from autogen import ConversableAgent, register_function

def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    # TODO
    # This function takes in a restaurant name and returns the reviews for that restaurant. 
    # The output should be a dictionary with the key being the restaurant name and the value being a list of reviews for that restaurant.
    # The "data fetch agent" should have access to this function signature, and it should be able to suggest this as a function call. 
    # Example:
    # > fetch_restaurant_data("Applebee's")
    # {"Applebee's": ["The food at Applebee's was average, with nothing particularly standing out.", ...]}
    file_path = "restaurant-data.txt"
    reviews = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                # normalize restaurant name (sometimes has "-" e.g. "In N Out" can be "In-N-Out"), take into accoutn case-insensitivity
                res_name = line.split(".")[0].strip().replace("-", " ").lower()
                print(res_name)
                if restaurant_name.lower() in res_name or restaurant_name.lower() in line.split(".")[0].lower():
                    reviews.append(line.split(".")[1].strip())

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {file_path} not found.")

    return {restaurant_name: reviews}


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    # TODO
    # This function takes in a restaurant name, a list of food scores from 1-5, and a list of customer service scores from 1-5
    # The output should be a score between 0 and 10, which is computed as the following:
    # SUM(sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    # The above formula is a geometric mean of the scores, which penalizes food quality more than customer service. 
    # Example:
    # > calculate_overall_score("Applebee's", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    # {"Applebee's": 5.048}
    # NOTE: be sure to that the score includes AT LEAST 3  decimal places. The public tests will only read scores that have 
    # at least 3 decimal places.
    # Compute overall score
    # Validate input
    if len(food_scores) != len(customer_service_scores):
        raise ValueError("The lengths of food_scores and customer_service_scores must be the same.")
    if not food_scores or not customer_service_scores:
        raise ValueError("The input lists cannot be empty.")
    
    # Compute overall score
    N = len(food_scores)
    score = sum([math.sqrt(food_scores[i]**2 * customer_service_scores[i]) for i in range(N)]) * 1/(N * math.sqrt(125)) * 10
    return {restaurant_name: round(score, 3)}

def get_data_fetch_agent_prompt(restaurant_query: str) -> str:
    # TODO
    # It may help to organize messages/prompts within a function which returns a string. 
    # For example, you could use this function to return a prompt for the data fetch agent 
    # to use to fetch reviews for a specific restaurant.
    return f"Please fetch the reviews for {restaurant_query} from the available dataset."

# TODO: feel free to write as many additional functions as you'd like.



# Do not modify the signature of the "main" function.
def main(user_query: str):
    entrypoint_agent_system_message = "You decide which agent to use based on the user query. You will delegate the task to the appropriate agent and aggregate the results. Ensure to fetch reviews, analyze scores, and calculate an overall score sequentially."
    # example LLM config for the entrypoint agent
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    # the main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent("entrypoint_agent", 
                                        system_message=entrypoint_agent_system_message, 
                                        llm_config=llm_config)
    
    data_fetch_agent = ConversableAgent(
        "data_fetch_agent",
        system_message="You fetch reviews for a restaurant based on the restaurant name in the query. Provide structured output: {restaurant_name: [list of reviews]}.",
        human_input_mode="NEVER",
        llm_config=llm_config
    )

    review_analysis_agent = ConversableAgent(name="review_analysis_agent",
                                        system_message="""You analyze reviews to extract food and customer service scores. Strictly map adjectives in the reviews to scores as described:
                                                        - "awful", "horrible", "disgusting" → 1/5
                                                        - "bad", "unpleasant", "offensive" → 2/5
                                                        - "average", "uninspiring", "forgettable" → 3/5
                                                        - "good", "enjoyable", "satisfying" → 4/5
                                                        - "awesome", "incredible", "amazing" → 5/5
                                                        Output a dictionary with the following keys: {restaurant_name, food_score_list, customer_service_score_list}.
                                                        Ensure that food_score_list and customer_service_score_list are of the same length.
                                                        """,
                                        llm_config=llm_config,
                                        human_input_mode="NEVER",
                                        max_consecutive_auto_reply=1)


    # Sub-agent for score calculation
    score_calculation_system_message = (
        "You calculate the overall score for a restaurant_name, food_scores, and customer_service_scores. "
        "Based on the calculation result, return in the format: the average food quality score is xx.xxx"
    )
    score_agent = ConversableAgent(
        "score_agent", 
        system_message=score_calculation_system_message, 
        human_input_mode="NEVER",
        llm_config=llm_config
    )


    data_fetch_agent.register_for_llm(name="fetch_restaurant_data", description="Fetches the reviews for a specific restaurant.")(fetch_restaurant_data)
    score_agent.register_for_llm(name="calculate_overall_score", description="calculate overall score")(calculate_overall_score)
    entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)

    # TODO
    # Fill in the argument to `initiate_chats` below, calling the correct agents sequentially.
    # If you decide to use another conversation pattern, feel free to disregard this code.
    result = entrypoint_agent.initiate_chats([
        {
            "recipient": data_fetch_agent,
            "message": user_query,
            "max_turns": 2,
            "summary_method": "last_msg"
        },
        {
            "recipient": review_analysis_agent,
            "message": "Here are the fetched reviews. Analyze them to extract scores for food and service.",
            "max_turns": 1,
            "summary_method": "reflection_with_llm",
            "summary_args": {
                "summary_prompt": "Provide a structured output: {restaurant_name, food_score_list, customer_service_score_list}."
            },
        },
        {
            "recipient": score_agent,
            "message": "Here are the scores extracted from reviews. Calculate the overall score.",
            "summary_method": "reflection_with_llm",
            "max_turns": 2,
        }
    ])
    
# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])

    # HAVING THIS ISSUE: https://github.com/microsoft/autogen/issues/3345