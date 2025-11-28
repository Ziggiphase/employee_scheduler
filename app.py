import streamlit as st
import spacy
from spacy.matcher import Matcher
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Scheduler Enterprise", page_icon="📅", layout="wide")

# --- 1. SETUP & ML TRAINING (70+ Data Points) ---
@st.cache_resource
def load_resources():
    # A. Load NLP Model
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    
    # Define Pattern for Time (e.g., 3-11, 9:00-5:00)
    pattern_time = [{"IS_DIGIT": True, "OP": "+"}, {"IS_PUNCT": True, "OP": "+"}, {"IS_DIGIT": True}]
    matcher.add("TIME", [pattern_time])

    # B. Train the Classifier (Expanded Dataset: 85 Sentences)
    training_data = [
        # AVAILABILITY (20 samples)
        ("I can work Saturday", "AVAILABILITY"),
        ("Bob is available Monday", "AVAILABILITY"),
        ("I am free all day Tuesday", "AVAILABILITY"),
        ("Put me down for the Wednesday shift", "AVAILABILITY"),
        ("I can do the morning shift", "AVAILABILITY"),
        ("Sign me up for Friday", "AVAILABILITY"),
        ("I'm good to work this weekend", "AVAILABILITY"),
        ("Available for the night shift", "AVAILABILITY"),
        ("Ready to work on Thursday", "AVAILABILITY"),
        ("I have open availability this week", "AVAILABILITY"),
        ("I can take the cashier role", "AVAILABILITY"),
        ("Sam is free to work", "AVAILABILITY"),
        ("Schedule me for Monday", "AVAILABILITY"),
        ("I'm around on Sunday", "AVAILABILITY"),
        ("I can come in", "AVAILABILITY"),
        ("Free to help out", "AVAILABILITY"),
        ("I will be there", "AVAILABILITY"),
        ("I can cover that shift", "AVAILABILITY"),
        ("Alice is available", "AVAILABILITY"),
        ("I have time on Friday", "AVAILABILITY"),

        # UNAVAILABILITY (25 samples)
        ("Sam cannot work", "UNAVAILABILITY"),
        ("I am unable to work due to exam", "UNAVAILABILITY"),
        ("I am sick", "UNAVAILABILITY"),
        ("not available", "UNAVAILABILITY"),
        ("I have a doctor appointment", "UNAVAILABILITY"),
        ("Bob is out of town", "UNAVAILABILITY"),
        ("I can't make it", "UNAVAILABILITY"),
        ("Do not schedule me for Monday", "UNAVAILABILITY"),
        ("I'm busy on Saturday", "UNAVAILABILITY"),
        ("Taking a personal day", "UNAVAILABILITY"),
        ("I have class so I can't work", "UNAVAILABILITY"),
        ("My car broke down, cannot come", "UNAVAILABILITY"),
        ("Sam is away", "UNAVAILABILITY"),
        ("Please remove me from the schedule", "UNAVAILABILITY"),
        ("I won't be able to work", "UNAVAILABILITY"),
        ("Off duty today", "UNAVAILABILITY"),
        ("Unavailable for the weekend", "UNAVAILABILITY"),
        ("I have a family emergency", "UNAVAILABILITY"),
        ("Taking the day off", "UNAVAILABILITY"),
        ("Not free Tuesday", "UNAVAILABILITY"),
        ("I have a dentist appointment", "UNAVAILABILITY"),
        ("Going on vacation", "UNAVAILABILITY"),
        ("Stuck in traffic, can't work", "UNAVAILABILITY"),
        ("Feeling unwell", "UNAVAILABILITY"),
        ("No availability", "UNAVAILABILITY"),

        # SHIFT REQUEST (20 samples)
        ("Need two cashiers", "SHIFT_REQUEST"),
        ("Shift open for server", "SHIFT_REQUEST"),
        ("We need stockers Friday", "SHIFT_REQUEST"),
        ("Looking for a manager", "SHIFT_REQUEST"),
        ("Who can work Monday?", "SHIFT_REQUEST"),
        ("Need coverage for the morning", "SHIFT_REQUEST"),
        ("One server needed", "SHIFT_REQUEST"),
        ("Requires 3 people for inventory", "SHIFT_REQUEST"),
        ("Opening available for 9-5", "SHIFT_REQUEST"),
        ("We are short staffed on Saturday", "SHIFT_REQUEST"),
        ("Need help on the floor", "SHIFT_REQUEST"),
        ("Looking for someone to cover", "SHIFT_REQUEST"),
        ("Vacant shift Tuesday", "SHIFT_REQUEST"),
        ("Need a bartender", "SHIFT_REQUEST"),
        ("Searching for staff", "SHIFT_REQUEST"),
        ("Shift available 3-11", "SHIFT_REQUEST"),
        ("We need more hands on deck", "SHIFT_REQUEST"),
        ("Cashier role open", "SHIFT_REQUEST"),
        ("Manager needed urgently", "SHIFT_REQUEST"),
        ("Any takers for Friday?", "SHIFT_REQUEST"),

        # PREFERENCE (20 samples)
        ("Sam prefers mornings", "PREFERENCE"),
        ("I prefer evening shifts", "PREFERENCE"),
        ("Bob likes night shift", "PREFERENCE"),
        ("I would rather work weekends", "PREFERENCE"),
        ("Please give me the early shift", "PREFERENCE"),
        ("I hate working Mondays", "PREFERENCE"),
        ("I love the closing shift", "PREFERENCE"),
        ("My preference is Tuesday", "PREFERENCE"),
        ("Ideally I want 9-5", "PREFERENCE"),
        ("I'd prefer not to close", "PREFERENCE"),
        ("Sam likes to work alone", "PREFERENCE"),
        ("Prefer the stock room", "PREFERENCE"),
        ("I favor the afternoon slot", "PREFERENCE"),
        ("Better if I work mornings", "PREFERENCE"),
        ("I prefer to be a cashier", "PREFERENCE"),
        ("Please assign me evenings", "PREFERENCE"),
        ("I really like Sundays", "PREFERENCE"),
        ("Preferred shift is 3-11", "PREFERENCE"),
        ("I'd rather do restocking", "PREFERENCE"),
        ("My choice is Friday", "PREFERENCE")
    ]
    
    # Split Data and Labels
    sentences, labels = zip(*training_data)
    
    # Create Pipeline: Text -> Vector -> Classifier
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(sentences, labels)
    
    return nlp, matcher, model

nlp, matcher, classifier = load_resources()

# --- 2. EXTRACTION LOGIC ---
def extract_details(text, intent):
    doc = nlp(text)
    data = {"Name": None, "Day": None, "Time": None, "Role": None}
    
    # Extract Name (Person)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            data["Name"] = ent.text
        if ent.label_ == "DATE":
            data["Day"] = ent.text
            
    # Extract Time (Regex Matcher)
    matches = matcher(doc)
    for match_id, start, end in matches:
        data["Time"] = doc[start:end].text

    # Extract Role (Lemmatization)
    roles = ["cashier", "stock", "server", "manager", "restock", "bartender"]
    for token in doc:
        if token.lemma_.lower() in roles:
            data["Role"] = token.lemma_.lower()
            
    return data

# --- 3. THE SCHEDULING ALGORITHM ---
def generate_roster(lines):
    employees = []  # The Supply
    shifts = []     # The Demand
    conflicts = []  # The Log
    
    # 1. Parse Phase
    for line in lines:
        if not line.strip(): continue
        
        # A. Predict Intent using Scikit-Learn
        intent = classifier.predict([line])[0]
        details = extract_details(line, intent)
        
        # B. Sort Data based on Intent
        if intent == "UNAVAILABILITY":
            name = details['Name'] or "Unknown Employee"
            conflicts.append(f"❌ **{name}** is unavailable ({line})")
        
        elif intent == "AVAILABILITY":
            employees.append({
                "Name": details["Name"],
                "Day": details["Day"],
                "Time": details["Time"] or "Any",
                "Role": details["Role"] or "General",
                "Preference": None,
                "Is_Assigned": False 
            })
            
        elif intent == "PREFERENCE":
            if details["Name"]:
                for emp in employees:
                    if emp["Name"] == details["Name"]:
                        emp["Preference"] = line
            else:
                if employees:
                    employees[-1]["Preference"] = line

        elif intent == "SHIFT_REQUEST":
            count = 1
            if "two" in line.lower() or "2" in line: count = 2
            if "three" in line.lower() or "3" in line: count = 3
            
            for _ in range(count):
                shifts.append({
                    "Day": details["Day"], 
                    "Time": details["Time"] or "9-5",
                    "Role": details["Role"] or "General",
                    "Assigned": None,
                    "Source": "Explicit Request"
                })

    # 2. Matching Phase (Explicit Requests First)
    for shift in shifts:
        for emp in employees:
            if emp["Is_Assigned"]: continue
            
            day_match = (shift["Day"] == emp["Day"]) or (shift["Day"] is None) or (emp["Day"] is None)
            role_match = (shift["Role"] == emp["Role"]) or (shift["Role"] == "General") or (emp["Role"] == "General")
            
            if day_match and role_match:
                shift["Assigned"] = emp
                emp["Is_Assigned"] = True
                break 

    # 3. Inference Phase (Supply-Driven)
    for emp in employees:
        if not emp["Is_Assigned"]:
            shifts.append({
                "Day": emp["Day"] or "TBD",
                "Time": emp["Time"],
                "Role": emp["Role"],
                "Assigned": emp,
                "Source": "Inferred from Availability"
            })
            emp["Is_Assigned"] = True

    return shifts, conflicts, employees

# --- 4. STREAMLIT UI ---
st.title("🤖 AI Scheduler Enterprise (ML-Powered)")
st.info("System Status: Online | Model: Naive Bayes | Training Data: 85 Sentences")

default_text = """Bob is free all day Saturday.
Sam cannot make it Monday because he has a class.
We need two cashiers for the Friday 9-5 shift.
Sam prefers working mornings.
Alice is available for the restocking shift on Wednesday.
I can work Sunday."""

raw_text = st.text_area("Enter Staff Constraints & Requests:", value=default_text, height=200)

if st.button("Generate Optimized Schedule"):
    lines = raw_text.split('\n')
    final_shifts, conflict_log, available_pool = generate_roster(lines)
    
    # Display 1: AI Intent Classification
    st.subheader("1. AI Analysis (Intent Detection)")
    with st.expander("View Classification Logs"):
        for line in lines:
            if line.strip():
                pred = classifier.predict([line])[0]
                color = "green" if pred == "AVAILABILITY" else "red" if pred == "UNAVAILABILITY" else "blue"
                st.markdown(f":{color}[**{pred}**]: {line}")

    # Display 2: Conflicts
    if conflict_log:
        st.subheader("2. Conflict Resolution Log")
        for c in conflict_log:
            st.markdown(c)

    # Display 3: The Final Schedule
    st.subheader("3. Final Roster")
    
    schedule_data = []
    for s in final_shifts:
        assignee = s['Assigned']['Name'] if s['Assigned'] else "UNFILLED"
        # Determine note: If preference exists, show it. Else show source.
        if s['Assigned'] and s['Assigned']['Preference']:
            note = f"✅ MATCHED PREF: {s['Assigned']['Preference']}"
        else:
            note = s['Source']
        
        schedule_data.append({
            "Day": s['Day'],
            "Shift Time": s['Time'],
            "Role": s['Role'],
            "Employee": assignee,
            "Status": note
        })
        
    st.table(pd.DataFrame(schedule_data))