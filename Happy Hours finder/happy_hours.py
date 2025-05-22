#!/usr/bin/env python3
"""
Happy Hour Finder AI Agent - Main Application

This is the main entry point for the Happy Hour Finder application,
orchestrating the different agents and utilities using LangGraph.
It allows users to find happy hour deals near their current location
(auto-detected), a specified address, or manually entered coordinates.
This version specifically enhances searching for happy hour deals at hotels.
"""

import argparse
import logging
import sys
import requests
import json
import os
import webbrowser
import socket
import ipaddress
from typing import Dict, List, Optional, Tuple, TypedDict
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from playsound import playsound # Import playsound

# LangGraph imports
from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

# --- IMPORTANT: Google Places API Key Configuration ---
# This script will first try to load the API key from the GOOGLE_PLACES_API_KEY
# environment variable (e.g., from your .env file). If it's not found there,
# it will fall back to the hardcoded value below.
# For production applications, always use environment variables for sensitive keys.
GOOGLE_PLACES_API_KEY = os.getenv('GOOGLE_PLACES_API_KEY', "YOUR_GOOGLE_PLACES_API_KEY_HERE")
# Replace "YOUR_GOOGLE_PLACES_API_KEY_HERE" with your actual Google Places API Key
# if you are not using a .env file.
# -----------------------------------------------------------------

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("happy_hour_finder")

# Suppress excessive logging from libraries to keep console clean
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("geopy").setLevel(logging.WARNING)


# --- 1. Define the LangGraph State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        lat (float): Latitude of the user's location.
        lng (float): Longitude of the user's location.
        happy_hour_deals (List[Dict]): List of dictionaries, each representing a happy hour deal.
        api_creation_status (str): Status message from the mock Whistle API creation.
        error_message (str): Any error message encountered during the workflow.
        location_name (str): Human-readable name of the user's location.
    """
    lat: float
    lng: float
    happy_hour_deals: List[Dict]
    api_creation_status: str
    error_message: str
    location_name: str


# --- Helper Functions for Location and Maps ---

def open_Maps(lat: float, lng: float):
    """
    Open Google Maps in the default web browser with the given coordinates.
    Uses a specific Google Maps URL format that should work in sandboxed environments.
    """
    # Using googleusercontent.com prefix for compatibility in some environments
    maps_url = f"https://www.google.com/maps/search/?api=1&query=?q={lat},{lng}"
    print(f"Opening location in Google Maps: {maps_url}")
    try:
        webbrowser.open(maps_url)
    except Exception as e:
        print(f"Failed to open browser: {e}. Please manually navigate to: {maps_url}")


def get_user_location_auto() -> Dict:
    """
    Automatically detects the user's approximate location using IP geolocation services.
    Falls back to a default location (New York City) if a public IP cannot be determined
    or if the geolocation service fails.

    Returns:
        Dict: A dictionary containing 'lat', 'lng', 'location_name', 'formatted_address',
              'city', 'state', and 'country' of the detected location.
    """
    print("🌐 Attempting to automatically detect your location...")

    ip_address = None
    try:
        # Try to get external IP using ipify API
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        response.raise_for_status()
        ip_address = response.json()['ip']
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not get external IP from ipify: {e}. Falling back to local IP.")
    except Exception as e:
        logger.warning(f"Unexpected error getting external IP: {e}. Falling back to local IP.")

    if not ip_address or (ipaddress.ip_address(ip_address).is_private if ip_address else True):
        # Fallback to Chennai if public IP cannot be determined, as per instruction
        print("❌ Could not detect a public IP address. Using Chennai, India as default location.")
        return {
            "lat": 13.067439,  # Latitude for Chennai, India
            "lng": 80.237617,  # Longitude for Chennai, India
            "location_name": "Chennai, Tamil Nadu, India (Default)",
            "formatted_address": "Chennai, Tamil Nadu, India",
            "city": "Chennai",
            "state": "Tamil Nadu",
            "country": "India"
        }

    # Use IP geolocation service (ipapi.co)
    try:
        response = requests.get(f'https://ipapi.co/{ip_address}/json/', timeout=10)
        response.raise_for_status()
        location_data = response.json()

        lat = location_data.get('latitude')
        lng = location_data.get('longitude')
        city = location_data.get('city', 'Unknown')
        region = location_data.get('region', 'Unknown')
        country = location_data.get('country_name', 'Unknown')

        location_info = {
            "lat": lat,
            "lng": lng,
            "location_name": f"{city}, {region}, {country}",
            "formatted_address": f"{city}, {region}, {country}",
            "city": city,
            "state": region,
            "country": country
        }

        print(f"✅ Location detected: {location_info['location_name']}")
        return location_info
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during IP geolocation ({ip_address}): {e}. Using default location.")
        return {
            "lat": 13.067439,
            "lng": 80.237617,
            "location_name": "Chennai, Tamil Nadu, India (Default)",
            "formatted_address": "Chennai, Tamil Nadu, India",
            "city": "Chennai",
            "state": "Tamil Nadu",
            "country": "India"
        }
    except Exception as e:
        logger.error(f"Error automatically detecting location ({ip_address}): {e}. Using default location.")
        return {
            "lat": 13.067439,
            "lng": 80.237617,
            "location_name": "Chennai, Tamil Nadu, India (Default)",
            "formatted_address": "Chennai, Tamil Nadu, India",
            "city": "Chennai",
            "state": "Tamil Nadu",
            "country": "India"
        }


def get_location_from_coordinates(lat: float, lng: float) -> Dict:
    """
    Gets detailed location information (address, city, state, country)
    based on provided latitude and longitude using Google Geocoding API.

    Args:
        lat (float): Latitude coordinate.
        lng (float): Longitude coordinate.

    Returns:
        Dict: A dictionary with 'lat', 'lng', 'formatted_address', 'city', 'state', 'country'.
    """
    try:
        print(f"🔍 Looking up location information for coordinates: {lat:.4f}, {lng:.4f}")

        response = requests.get(
            f'https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_PLACES_API_KEY}',
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        location_info = {
            "lat": lat,
            "lng": lng,
            "formatted_address": "Unknown location",
            "city": "Unknown",
            "state": "Unknown",
            "country": "Unknown"
        }

        if data["status"] == "OK" and data.get("results"):
            result = data["results"][0]
            location_info["formatted_address"] = result.get("formatted_address", "Unknown location")

            for component in result.get("address_components", []):
                types = component.get("types", [])
                if "locality" in types:
                    location_info["city"] = component.get("long_name", "Unknown")
                elif "administrative_area_level_1" in types:
                    location_info["state"] = component.get("long_name", "Unknown")
                elif "country" in types:
                    location_info["country"] = component.get("long_name", "Unknown")

        print(f"✅ Location found: {location_info['formatted_address']}")
        return location_info
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error getting location information: {e}")
        return {"lat": lat, "lng": lng, "formatted_address": "Location lookup failed (Network Error)", "city": "Unknown", "state": "Unknown", "country": "Unknown"}
    except Exception as e:
        logger.error(f"Error getting location information: {e}")
        return {"lat": lat, "lng": lng, "formatted_address": "Location lookup failed", "city": "Unknown", "state": "Unknown", "country": "Unknown"}


def calculate_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculates the geodesic distance in kilometers between two latitude/longitude points
    using the `geopy` library for accuracy.
    """
    try:
        return geodesic((lat1, lng1), (lat2, lng2)).kilometers
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return float('inf') # Return a very large number on error


# --- Location Input Service (User Interaction) ---
class LocationInputService:
    """Service to handle user location input through various methods."""

    def __init__(self):
        self.geocoder = Nominatim(user_agent="happy_hour_finder_app")

    def get_user_location(self) -> Tuple[float, float, str]:
        """
        Guides the user through choosing a location input method (address, manual coords, auto-detect).

        Returns:
            Tuple[float, float, str]: (latitude, longitude, human-readable_location_name)
        """
        print("\n🌍 How would you like to provide your location?")
        print("1. Enter address/city name (we'll find coordinates)")
        print("2. Enter latitude and longitude manually")
        print("3. Use my current location (requires API/GPS)")

        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()

                if choice == "1":
                    return self._get_location_from_address()
                elif choice == "2":
                    return self._get_manual_coordinates()
                elif choice == "3":
                    # --- CRITICAL: Auto-detect and open map here ---
                    location_info = get_user_location_auto()
                    lat, lng = location_info['lat'], location_info['lng']
                    location_name = location_info['location_name']

                    if lat is None or lng is None:
                        print("❌ Failed to get valid coordinates from automatic detection. Please try again or choose another option.")
                        continue # Loop back to choice menu

                    # Open Google Maps with the detected location
                    open_Maps(lat, lng)
                    
                    # Confirm with user if this location is correct
                    print(f"\n📍 Detected location: {location_name} ({lat:.4f}, {lng:.4f})")
                    confirm = input("Is this your desired location for happy hour search? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return lat, lng, location_name
                    else:
                        print("Okay, let's try again or choose a different method.")
                        continue # Loop back to choice menu
                else:
                    print("❌ Please enter 1, 2, or 3")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Error in location input service: {e}", exc_info=True)
                print("An error occurred during location input. Please try again.")

    def _get_location_from_address(self) -> Tuple[float, float, str]:
        """
        Prompts the user for an address or city name and uses Nominatim to geocode it.
        """
        while True:
            address = input("\n📍 Enter your address or city name: ").strip()
            if not address:
                print("❌ Please enter a valid address.")
                continue

            try:
                print(f"🔍 Looking up coordinates for: {address}")
                location = self.geocoder.geocode(address, timeout=10)

                if location:
                    lat, lng = location.latitude, location.longitude
                    location_name = location.address
                    print(f"✅ Found coordinates: {lat:.4f}, {lng:.4f}")
                    print(f"📍 Location: {location_name}")

                    confirm = input("Is this correct? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return lat, lng, location_name
                    else:
                        continue
                else:
                    print("❌ Could not find coordinates for that address. Please try again.")

            except Exception as e:
                logger.error(f"Error looking up address via Nominatim: {e}", exc_info=True)
                print("An error occurred during address lookup. Please try again with a different address.")

    def _get_manual_coordinates(self) -> Tuple[float, float, str]:
        """
        Prompts the user to manually enter latitude and longitude.
        Attempts to reverse geocode for confirmation.
        """
        while True:
            try:
                lat_input = input("\n📍 Enter latitude (e.g., 40.7128): ").strip()
                lng_input = input("📍 Enter longitude (e.g., -74.0060): ").strip()

                lat = float(lat_input)
                lng = float(lng_input)

                # Basic validation
                if not (-90 <= lat <= 90):
                    print("❌ Latitude must be between -90 and 90.")
                    continue
                if not (-180 <= lng <= 180):
                    print("❌ Longitude must be between -180 and 180.")
                    continue

                # Try to get address for confirmation
                location_name = f"Manually entered coordinates: {lat:.4f}, {lng:.4f}" # Default if reverse fails
                try:
                    location = self.geocoder.reverse(f"{lat}, {lng}", timeout=10)
                    if location:
                        location_name = location.address
                        print(f"📍 This appears to be: {location_name}")
                except Exception as e:
                    logger.warning(f"Could not reverse geocode manual coordinates: {e}. Displaying raw coordinates.")
                    print(f"📍 Coordinates: {lat:.4f}, {lng:.4f}")

                confirm = input("Are these coordinates correct? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return lat, lng, location_name

            except ValueError:
                print("❌ Please enter valid numbers for latitude and longitude.")
            except Exception as e:
                logger.error(f"Error in manual coordinate input: {e}", exc_info=True)
                print("An error occurred during manual coordinate input. Please try again.")


# --- Live External Functions (Using Google Places API for Search) ---

def fetch_happy_hour_data_live(lat: float, lng: float, radius_km: float = 5.0) -> List[Dict]:
    """
    Fetches potential happy hour venues from Google Places API using Text Search and Details API.
    It performs multiple searches with relevant keywords and then enriches the data with details
    like reviews, phone numbers, and websites. It also applies a heuristic to infer happy hour presence.

    Args:
        lat (float): Latitude of the center point for the search.
        lng (float): Longitude of the center point for the search.
        radius_km (float): Search radius in kilometers.

    Returns:
        List[Dict]: A list of dictionaries, each representing a potential happy hour venue
                    with enriched details and inferred happy hour information.
    """
    if not GOOGLE_PLACES_API_KEY or GOOGLE_PLACES_API_KEY == "YOUR_GOOGLE_PLACES_API_KEY_HERE":
        raise ValueError(
            "Google Places API Key is not set or is invalid. "
            "Please set GOOGLE_PLACES_API_KEY in your .env file or replace "
            "'YOUR_GOOGLE_PLACES_API_KEY_HERE' in the script."
        )

    logger.info(f"Fetching live data for lat={lat:.4f}, lng={lng:.4f} within {radius_km}km using Google Places API.")

    radius_meters = int(radius_km * 1000) # Google Places API uses radius in meters

    # Keywords to search for potential happy hour places, explicitly including hotels
    # Prioritizing hotel-related terms for your specific request
    search_queries = [
        "hotel happy hour", "hotel bar", "hotel lounge", "lodging happy hour",
        "bar", "pub", "lounge", "brewery", "restaurant with happy hour",
        "cocktail bar", "wine bar"
    ]

    all_places_found = []
    place_ids_found = set() # To prevent adding duplicate places (identified by place_id)

    for query_term in search_queries:
        params = {
            "query": query_term,
            "location": f"{lat},{lng}",
            "radius": radius_meters,
            "key": GOOGLE_PLACES_API_KEY
        }

        try:
            response = requests.get("https://maps.googleapis.com/maps/api/place/textsearch/json", params=params, timeout=15)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if data["status"] == "OK":
                for place in data["results"]:
                    place_id = place.get("place_id")
                    if place_id and place_id not in place_ids_found:
                        if "geometry" in place and "location" in place["geometry"]:
                            place_lat = place["geometry"]["location"]["lat"]
                            place_lng = place["geometry"]["location"]["lng"]

                            # Get additional details for opening hours, phone, website, reviews
                            # This is crucial for inferring happy hour presence
                            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                            details_params = {
                                'place_id': place_id,
                                'fields': 'name,formatted_address,geometry,rating,opening_hours,website,formatted_phone_number,reviews,types',
                                'key': GOOGLE_PLACES_API_KEY
                            }
                            details_response = requests.get(details_url, params=details_params, timeout=10)
                            details_response.raise_for_status()
                            details_data = details_response.json()
                            
                            # Use detailed place info if available, otherwise fallback to basic place info
                            detailed_place = details_data.get('result', {}) if details_data.get('status') == 'OK' else place

                            # Heuristic for happy hour: Check reviews for keywords
                            has_happy_hour_mention = False
                            reviews = detailed_place.get('reviews', [])
                            for review in reviews[:5]: # Check up to 5 most recent reviews
                                text = review.get('text', '').lower()
                                if any(keyword in text for keyword in ['happy hour', 'drink special', 'half price', 'discount', 'specials', 'deal']):
                                    has_happy_hour_mention = True
                                    break
                            
                            # Determine offer details and validity based on type and review mentions
                            offer_details = f"Potential happy hour at {detailed_place.get('name', 'Unknown Place')}."
                            offer_validity_hours = 2 # Default to 2 hours

                            place_types = detailed_place.get('types', [])
                            # Enhanced logic for hotel-specific offers
                            if 'lodging' in place_types or 'hotel' in place_types:
                                offer_details = "Hotel Lobby Bar Happy Hour: Special deals on drinks and appetizers for guests and visitors."
                                offer_validity_hours = 2 # Hotels might have slightly longer or earlier happy hours
                                if 'bar' in place_types or 'restaurant' in place_types:
                                    offer_details = "Hotel Bar/Restaurant Happy Hour: Discounted cocktails, wine, beer, and perhaps small plates."
                                    offer_validity_hours = 2.5
                            elif 'bar' in place_types or 'pub' in place_types or 'night_club' in place_types:
                                offer_details = "Happy Hour: Discounted drinks and possibly appetizers."
                                offer_validity_hours = 3
                            elif 'restaurant' in place_types:
                                offer_details = "Happy Hour: Special deals on drinks and appetizers."
                                offer_validity_hours = 2

                            if has_happy_hour_mention:
                                # If happy hour is explicitly mentioned, make the offer more confident
                                offer_details = f"Happy hour confirmed in recent reviews! Deals likely include: {offer_details.split(':')[1].strip() if ':' in offer_details else 'discounted drinks/food'}. Call to confirm current offers."
                                offer_validity_hours = max(offer_validity_hours, 2) # Ensure at least 2 hours if mentioned

                            all_places_found.append({
                                "name": detailed_place.get("name"),
                                "lat": place_lat,
                                "lng": place_lng,
                                "place_id": place_id,
                                "types": detailed_place.get("types", []),
                                "formatted_address": detailed_place.get("formatted_address", place.get("vicinity", "Address not available")),
                                "phone": detailed_place.get("formatted_phone_number", "N/A"),
                                "website": detailed_place.get("website", "N/A"),
                                "rating": detailed_place.get("rating"),
                                "opening_hours": detailed_place.get("opening_hours", {}).get('weekday_text', []),
                                "details": offer_details,
                                "validity_hours": offer_validity_hours
                            })
                            place_ids_found.add(place_id)
            elif data["status"] == "ZERO_RESULTS":
                logger.info(f"No results found for query: '{query_term}'.")
            else:
                logger.error(f"Google Places API status for '{query_term}': {data['status']} - {data.get('error_message', 'No error message provided.')}")
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed for '{query_term}': {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response for '{query_term}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for '{query_term}': {e}", exc_info=True)

    # Filter by distance more strictly after collecting all potential places
    filtered_deals = []
    for place in all_places_found:
        distance = calculate_distance_km(lat, lng, place["lat"], place["lng"])
        if distance <= radius_km:
            place['distance_km'] = distance # Add distance for display
            filtered_deals.append(place)

    logger.info(f"Google Places API found {len(filtered_deals)} places within {radius_km}km after filtering.")
    return filtered_deals


# --- Mock Whistle API Function ---
def create_whistle_api_mock(deals_json: List[Dict]) -> str:
    """
    MOCK function to simulate creating an API endpoint via Whistle.
    In a real scenario, this would make an HTTP POST request to the Whistle API
    to publish the happy hour deals.
    Plays a sound to indicate creation.
    """
    print("\nDEBUG: Mocking Whistle API creation...")
    # Uncomment the line below to see the full JSON data being "published"
    # print(f"DEBUG: Data received for API creation: {json.dumps(deals_json, indent=2)}")
    
    if deals_json:
        # Define the path to your sound file
        # Make sure 'whistle.mp3' is in the same directory as this script
        sound_file = "whistle1.mp3"
        if os.path.exists(sound_file):
            try:
                playsound(sound_file)
                print("🎶 Whistle sound played!")
            except Exception as e:
                print(f"⚠️ Could not play sound: {e}. Ensure 'playsound' dependencies are met and the file is valid.")
        else:
            print(f"⚠️ Sound file '{sound_file}' not found. No sound will be played.")

        return "Whistle API successfully created/updated with new happy hour deals."
    else:
        return "Whistle API creation failed: No deals to publish."


# --- 2. LangGraph Nodes (Agents/Functions) ---

def data_collector_agent(state: GraphState) -> GraphState:
    """
    LangGraph Agent Node: Collects nearby happy hour deals using the live data fetcher.
    Updates the 'happy_hour_deals' and 'error_message' in the graph state.
    """
    print("\n--- Running Data Collector Agent ---")
    lat = state.get("lat")
    lng = state.get("lng")
    radius = 5.0 # Fixed radius for this agent's search, can be made dynamic if needed

    if lat is None or lng is None:
        state["error_message"] = "Latitude and Longitude are required inputs for data collection."
        return state

    try:
        # Call the live data fetch function
        raw_deals = fetch_happy_hour_data_live(lat=lat, lng=lng, radius_km=radius)

        formatted_deals = []
        for deal in raw_deals:
            # Ensure offer_validity_hours is a positive number
            offer_validity = max(1, deal.get("validity_hours", 1))

            # Determine venue type for display, prioritizing 'Hotel'
            venue_type_display = "Venue"
            if deal.get('types'):
                if 'lodging' in deal['types'] or 'hotel' in deal['types']: venue_type_display = 'Hotel'
                elif 'bar' in deal['types']: venue_type_display = 'Bar'
                elif 'restaurant' in deal['types']: venue_type_display = 'Restaurant'
                elif 'cafe' in deal['types']: venue_type_display = 'Cafe'
                else: venue_type_display = deal['types'][0].replace('_', ' ').title() # Fallback to first type

            formatted_deals.append({
                "store_name": deal.get("name", "Unknown Store"),
                "store_address": {
                    "lat": deal.get("lat"),
                    "lng": deal.get("lng")
                },
                "offer_details": deal.get("details", "No specific offer details."),
                "offer_validity_hours": offer_validity,
                "alert_radius_km": 1,   # Fixed constraint for alert radius
                "provider": True
            })
        
        # Sort by distance immediately for consistency
        sorted_formatted_deals = sorted(formatted_deals, key=lambda x: calculate_distance_km(lat, lng, x['store_address']['lat'], x['store_address']['lng']))

        print(f"Data Collector found {len(sorted_formatted_deals)} happy hour deals.")
        state["happy_hour_deals"] = sorted_formatted_deals
        return state
    except Exception as e:
        state["error_message"] = f"Error in Data Collector: {e}"
        logger.error(f"ERROR in data_collector_agent: {e}", exc_info=True)
        return state

def whistle_api_creator_agent(state: GraphState) -> GraphState:
    """
    LangGraph Agent Node: Simulates creating/updating an API endpoint with the collected deals.
    Updates the 'api_creation_status' and 'error_message' in the graph state.
    """
    print("\n--- Running Whistle API Creator Agent ---")
    happy_hour_deals = state.get("happy_hour_deals", [])

    if not happy_hour_deals:
        print("No happy hour deals to publish. Skipping Whistle API creation.")
        state["api_creation_status"] = "No deals found to publish to Whistle API."
        return state

    try:
        status_message = create_whistle_api_mock(happy_hour_deals) # Call the mock API function
        state["api_creation_status"] = status_message
        print(f"Whistle API Creator status: {status_message}")
        return state
    except Exception as e:
        state["error_message"] = f"Error in Whistle API Creator: {e}"
        logger.error(f"ERROR in whistle_api_creator_agent: {e}", exc_info=True)
        return state

# --- 3. Define LangGraph Conditional Edge ---
def decide_next_step(state: GraphState) -> str:
    """
    Conditional logic to determine the next step in the LangGraph workflow.
    If data collection resulted in an error or no deals, the workflow ends.
    Otherwise, it proceeds to the Whistle API Creator.
    """
    if state.get("error_message"):
        print("Decision: Error encountered, ending process.")
        return "end_with_error"
    elif not state.get("happy_hour_deals"):
        print("Decision: No happy hour deals found, ending process.")
        return "end_no_deals"
    else:
        print("Decision: Happy hour deals found, proceeding to Whistle API Creator.")
        return "continue_to_whistle_api"

# --- 4. Build the LangGraph Workflow ---
workflow = StateGraph(GraphState)
workflow.add_node("data_collector", data_collector_agent)
workflow.add_node("whistle_api_creator", whistle_api_creator_agent)

# Set the entry point for the graph
workflow.set_entry_point("data_collector")

# Define conditional edges from the data_collector node
workflow.add_conditional_edges(
    "data_collector",
    decide_next_step,
    {
        "continue_to_whistle_api": "whistle_api_creator",
        "end_no_deals": END,
        "end_with_error": END
    }
)

# Define the edge from whistle_api_creator to the end of the graph
workflow.add_edge("whistle_api_creator", END)

# Compile the workflow into a runnable application
app = workflow.compile()


# --- 5. Main Application Execution ---

def main():
    """
    Main entry point for the Happy Hour Finder application.
    Handles command-line arguments and orchestrates the LangGraph workflow.
    """
    parser = argparse.ArgumentParser(
        description="Happy Hour Finder AI Agent - Find happy hour deals near you."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically detect location and find deals."
    )
    parser.add_argument(
        "--address",
        type=str,
        help="Specify an address (e.g., '1600 Amphitheatre Pkwy, Mountain View, CA')."
    )
    parser.add_argument(
        "--coords",
        type=str,
        help="Specify latitude and longitude (e.g., '34.0522,-118.2437')."
    )
    parser.add_argument(
        "--test_sound",
        action="store_true",
        help="Play the whistle sound to test audio output."
    )

    args = parser.parse_args()
    
    if args.test_sound:
        script_dir = os.path.dirname(__file__)
        # Corrected path for the sound file
        sound_file = os.path.join(script_dir, "pro11", "whistle1.mp3") 
        if os.path.exists(sound_file):
            try:
                print("Attempting to play whistle sound...")
                playsound(sound_file)
                print("Whistle sound test complete.")
            except Exception as e:
                print(f"Error playing sound: {e}. Please ensure 'playsound' is installed and your audio setup is working.")
        else:
            print(f"Sound file not found at: {sound_file}")
        sys.exit(0)

    location_service = LocationInputService()
    lat, lng, location_name = None, None, "Unknown Location"

    if args.auto:
        loc_info = get_user_location_auto()
        lat, lng = loc_info['lat'], loc_info['lng']
        location_name = loc_info['location_name']
    elif args.address:
        print(f"Searching for address: {args.address}")
        try:
            location = location_service.geocoder.geocode(args.address, timeout=10)
            if location:
                lat, lng = location.latitude, location.longitude
                location_name = location.address
                print(f"Found coordinates for '{args.address}': {lat:.4f}, {lng:.4f}")
            else:
                print(f"Could not find coordinates for address: {args.address}. Exiting.")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error geocoding address: {e}", exc_info=True)
            print(f"Error processing address '{args.address}'. Exiting.")
            sys.exit(1)
    elif args.coords:
        try:
            lat_str, lng_str = args.coords.split(',')
            lat, lng = float(lat_str), float(lng_str)
            loc_info = get_location_from_coordinates(lat, lng)
            location_name = loc_info['formatted_address']
            print(f"Using manual coordinates: {lat:.4f}, {lng:.4f} ({location_name})")
        except ValueError:
            print("Invalid coordinates format. Please use 'latitude,longitude' (e.g., '34.0522,-118.2437'). Exiting.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error processing coordinates: {e}", exc_info=True)
            print(f"Error processing coordinates '{args.coords}'. Exiting.")
            sys.exit(1)
    else:
        # No command-line arguments, prompt the user
        lat, lng, location_name = location_service.get_user_location()
        if lat is None or lng is None:
            print("Failed to get location. Exiting.")
            sys.exit(1)

    print(f"\n🚀 Starting Happy Hour search for {location_name} ({lat:.4f}, {lng:.4f})...")

    initial_state = GraphState(
        lat=lat,
        lng=lng,
        happy_hour_deals=[],
        api_creation_status="",
        error_message="",
        location_name=location_name
    )

    try:
        # Run the LangGraph workflow
        final_state = app.invoke(initial_state)

        print("\n--- Happy Hour Finder Results ---")
        if final_state["error_message"]:
            print(f"An error occurred: {final_state['error_message']}")
        elif final_state["happy_hour_deals"]:
            print(f"🎉 Found {len(final_state['happy_hour_deals'])} potential happy hour deals near {final_state['location_name']}:")
            for i, deal in enumerate(final_state["happy_hour_deals"]):
                print(json.dumps(deal, indent=2)) # Print in JSON format as requested
                # You can remove the following lines if you only want JSON output
                # distance_str = f"({calculate_distance_km(lat, lng, deal['store_address']['lat'], deal['store_address']['lng']):.2f} km away)" 
                # print(f"\n{i+1}. {deal.get('store_name')} {distance_str}")
                # print(f"   Offer: {deal.get('offer_details')}")
                # print(f"   Likely Duration: ~{deal.get('offer_validity_hours')} hours")

            print(f"\nWhistle API Status: {final_state['api_creation_status']}")
            print("\nTip: Call ahead to confirm happy hour times and offers!")
        else:
            print(f"😞 No happy hour deals found near {final_state['location_name']} in the selected radius. Try a different location or time!")
        
        # Always offer to open maps for the searched location
        if lat is not None and lng is not None:
            open_Maps(lat, lng)

    except Exception as e:
        logger.critical(f"Unhandled error during workflow execution: {e}", exc_info=True)
        print(f"\nAn unrecoverable error occurred: {e}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()