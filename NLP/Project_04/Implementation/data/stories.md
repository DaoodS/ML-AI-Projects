## complete path
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - slot{"location": "delhi"}
    - utter_ask_budget
* restaurant_search("budget": "300-700")
    - slot("budget":"300-700")
    - action_search_restaurants
    - utter_ask_email
* action_send_email
    - action_send_email("email":"email")
    - utter_sent
* affirm
    - utter_goodbye
    - export


## location specified
* greet
    - utter_greet
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_budget
* restaurant_search("budget": "300-700")
    - slot("budget":"300-700")
    - action_search_restaurants
* action_send_email
    - utter_ask_email
* action_send_email
    - action_send_email("email":"email")
    - utter_sent
* affirm
    - utter_goodbye
    - export


## complete path 2
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "North Indian"}
    - slot{"cuisine": "North Indian"}
    - utter_ask_budget
* restaurant_search("budget": "300-700")
    - slot("budget":"300-700")
    - action_search_restaurants
* action_send_email
    - utter_ask_email
* action_send_email
    - action_send_email("email":"email")
    - utter_sent
    - utter_goodbye


## complete path 3
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "italy"}
    - slot{"location": "italy"}
	- utter_ask_cuisine
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_budget
* restaurant_search("budget": "300-700")
    - slot("budget":"300-700")
    - action_search_restaurants
* goodbye
    - utter_goodbye


## complete path 4
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_budget
* restaurant_search("budget": "300-700")
    - slot("budget":"300-700")
    - action_search_restaurants
    - slot{"location": "delhi"}
    - export


## interactive_story_1
* greet
    - utter_greet
* restaurant_search{"location": "mumbai"}
    - slot{"location": "mumbai"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_budget
* restaurant_search("budget": "300-700")
    - slot("budget":"300-700")
    - action_search_restaurants
    - slot{"location": "mumbai"}
* stop


## interactive_story_1
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "mumbai"}
    - slot{"location": "mumbai"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Italian"}
    - slot{"cuisine": "Italian"}
    - utter_ask_budget
* restaurant_search("budget": "300-700")
    - slot("budget":"300-700")
    - action_search_restaurants
    - slot{"location": "mumbai"}


## interactive_story_1
* greet
    - utter_greet
* restaurant_search{"cuisine": "chinese", "location": "delhi"}
    - slot{"cuisine": "chinese"}
    - slot{"location": "delhi"}
    - action_search_restaurants
    - slot{"location": "delhi"}
    - slot("budget":"300-700")
* affirm
    - utter_goodbye


## interactive_story_1
* greet
    - utter_greet
* restaurant_search{"cuisine": "chinese", "location": "delhi"}
    - slot{"cuisine": "chinese"}
    - slot{"location": "delhi"}
    - utter_ask_budget
* restaurant_search("budget": "300-700")
    - slot("budget":"300-700")
    - action_search_restaurants
    - slot{"location": "delhi"}
* affirm
    - utter_goodbye

    
    
## happy_path
* greet
    - utter_greet
* restaurant_search{"cuisine": "italian", "location": "mumbai"}
    - slot{"cuisine": "italian"}
    - slot{"location": "mumbai"}
    - action_search_restaurants
    - slot{"location": "mumbai"}
* affirm
    - utter_goodbye


## interactive_story_1
* greet
    - utter_greet
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_search_restaurants
    - slot{"location": "delhi"}
* affirm
    - utter_goodbye
