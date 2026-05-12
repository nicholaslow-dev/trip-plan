# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Eval Script for Trip Planning."""

import json
import re
from typing import Any

from absl import app
from absl import flags


_DATA_PATH = flags.DEFINE_string(
    'data_path',
    'data/trip_planning.json',
    'path to the data file containing model responses in json format.',
)

_RESPONSE_KEY = flags.DEFINE_string(
    'response_key',
    'pred_5shot_pro',
    'The key in the JSON data that contains the model prediction.',
)

_ERROR_FILE_PATH = flags.DEFINE_string(
    'error_file_path',
    'data/trip_planning_errors.json',
    'Path to save the error analysis JSON file.',
)


def parse_response(response: str):
  """Parse the response.

  Returns a parsed plan in a list of (city, stay_days) tuples.

  Args:
    response: Raw response from the model.

  Returns:
    Structured plan after parsing.
  """
  # Remove <think>...</think> blocks if present (common in reasoning models)
  response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

  pattern_visit = r'\d+-\d+'
  pattern_flight = r'.*Day (\d+).*from (\w+) to (\w+)'
  pattern_days = r'European cities for (\d+) days'

  days, flights, flight_days = [], [], []
  total_days = None
  for piece in response.split('\n'):
    days_match = re.findall(pattern_days, piece)
    if days_match:
      total_days = int(days_match[0])

    # Match "Day X-Y" or "Day X"
    visit_match = re.search(r'\*\*Day (\d+)(?:-(\d+))?:\*\*', piece)
    if visit_match:
      start_day = visit_match.group(1)
      end_day = visit_match.group(2) if visit_match.group(2) else start_day
      days.append(f"{start_day}-{end_day}")
      
      # Reach the end of the plan, stop to avoid parsing alternative plans.
      if total_days and int(end_day) == total_days:
        flight_match = re.search(pattern_flight, piece) # Check for flight on last day line
        if flight_match:
             flights.append(flight_match.groups())
        break
    
    flight_match = re.search(pattern_flight, piece)
    if flight_match:
      flights.append(flight_match.groups())

  visit_cities, parsed_plan = [], []
  for flight_day, begin_city, end_city in flights:
    flight_days.append(int(flight_day))
    if not visit_cities:
      visit_cities.append(begin_city)
      visit_cities.append(end_city)
    else:
      visit_cities.append(end_city)

  if not days or not flights or not visit_cities:
    return []
  last_day = int(days[-1].split('-')[1])
  flight_days = [1] + flight_days + [last_day]
  for i, visit_city in enumerate(visit_cities):
    city_stay = flight_days[i + 1] - flight_days[i] + 1
    parsed_plan.append((visit_city, city_stay))

  return parsed_plan


def compute_example_score(cities: str, durations: str, parsed_plan: list[Any]):
  """Compute the exact-match accuracy.

  Compute the example-level exact_match score (0/1) given the parsed plan
  and the ground truth in the format of durations and cities.

  Args:
    cities: The cities in the plan in the format of "city1**city2**city3".
    durations: The durations of the stay in each city in the format of
      "1**2**3".
    parsed_plan: The parsed plan from the response.

  Returns:
    Exact-match accuracy of 0 (mismatched) or 1 (matched).
  """

  stays = [x for x in cities.split('**') if x]
  days = [int(x) for x in durations.split('**') if x]
  num_stays = min(len(stays), len(parsed_plan))
  num_match = 0
  for i in range(num_stays):
    if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
      num_match += 1
    else:
      break
  hard_score = 0.0 if num_match / len(stays) < 1.0 else 1.0
  return hard_score


def compute_score(
    cities: list[str], durations: list[str], responses: list[str]
):
  """Compute the sample-level exact-match accuracy.

  Args:
    cities: List of cities in the plan in the format of "city1**city2**city3".
    durations: List of durations of the stay in each city in the format of
      "1**2**3".
    responses: The raw responses from the model.

  Returns:
    Exact-match score at the sample level.
  """
  parsed_plans = [parse_response(response) for response in responses]
  hard_scores = [
      compute_example_score(city, duration, parsed_plan)
      for city, duration, parsed_plan in zip(cities, durations, parsed_plans)
  ]
  hard_acc = sum(hard_scores) / len(hard_scores)
  return hard_acc


def main(_):
  with open(_DATA_PATH.value) as f:
    data = json.load(f)

  errors = {}
  total_score = 0
  sample_count = 0

  for key, item in data.items():
    cities = item['cities']
    durations = item['durations']
    durations = item['durations'] # Remove duplicate line if present or just keep one
    response = item.get(_RESPONSE_KEY.value, '')
    
    parsed_plan = parse_response(response)
    score = compute_example_score(cities, durations, parsed_plan)

    item['parsed_plan'] = parsed_plan
    item['score'] = score
    
    total_score += score
    sample_count += 1
    
    if score < 1.0:
        # Save the error details
        errors[key] = item.copy()

  hard_acc = total_score / sample_count if sample_count > 0 else 0
  print(f'EM Accuracy of {sample_count} samples: {hard_acc}')
  
  def save_json_with_compact_plan(json_data, file_path):
    """Saves JSON data with 'parsed_plan' fields compacted."""
    data_to_dump = {}
    mapping = {}
    counter = 0

    for k, v in json_data.items():
      # Create a shallow copy of the item
      item_copy = v.copy()
      if 'parsed_plan' in item_copy:
        placeholder = f"@@COMPACT_PLAN_{counter}@@"
        mapping[placeholder] = item_copy['parsed_plan']
        item_copy['parsed_plan'] = placeholder
        counter += 1
      data_to_dump[k] = item_copy

    # Dump with placeholders
    json_str = json.dumps(data_to_dump, indent=4)

    # Replace placeholders with compact JSON representation
    for placeholder, plan_data in mapping.items():
      compact_plan = json.dumps(plan_data)
      json_str = json_str.replace(f'"{placeholder}"', compact_plan)

    with open(file_path, 'w') as f:
      f.write(json_str)

  # Update the original data file with scores and parsed plans
  save_json_with_compact_plan(data, _DATA_PATH.value)
  print(f'Updated {_DATA_PATH.value} with scores and parsed plans.')
  
  if errors:
      error_file_path = _ERROR_FILE_PATH.value
      save_json_with_compact_plan(errors, error_file_path)
      print(f'Saved {len(errors)} errors to {error_file_path}')


if __name__ == '__main__':
  app.run(main)
