#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#In[5.1]
import streamlit as st

# Create a toggle button for Dark/Light mode
dark_mode = st.toggle("Dark Mode", value=False)

# Apply CSS styles dynamically
if dark_mode:
    st.markdown(
        """
        <style>
            /* Change background & text color */
            body { background-color: #070b1d; color: white; }
            .stApp { background-color: #070b1d; }
            
            /* Move the toggle to the top right */
            div[data-testid="stHorizontalBlock"] {
                position: absolute;
                top: 10px;
                right: 20px;
                z-index: 9999;
            }
            
            /* Change text to white */
            h1, h2, h3, h4, h5, h6, p, label { color: white !important; }
            
            /* Change text input & file upload background */
            .stTextInput>div>div>input, 
            .stTextArea>div>textarea, 
            .stFileUploader>div>div>button {
                background-color: #2a2d3a !important;
                color: white !important;
                border-radius: 5px;
                border: 1px solid #444 !important;
            }


            /* Button styles */
            .stButton>button { background-color: #0071ff; color: white; border-radius: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
            /* Light mode styling */
            body { background-color: white; color: black; }
            .stApp { background-color: white; }

            /* Move the toggle to the top right */
            div[data-testid="stHorizontalBlock"] {
                position: absolute;
                top: 10px;
                right: 20px;
                z-index: 9999;
            }
            
            /* Ensure text remains black */
            h1, h2, h3, h4, h5, h6, p, label { color: black !important; }
            
            /* Text input & file upload in light mode */
            .stTextInput>div>div>input, 
            .stTextArea>div>textarea, 
            .stFileUploader>div>div>button {
                background-color: #e8e9ed !important;
                color: black !important;
                border-radius: 5px;
                border: 1px solid #ccc !important;
            }

            /* Button styles */
            .stButton>button { background-color: #3a435d; color: white; border-radius: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )




# In[6]:


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text


# In[7]:


# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors [1:]
    cosine_similarities = cosine_similarity ([job_description_vector], resume_vectors).flatten()

    return cosine_similarities


# In[8]:


# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")



# Job description input
st.header("Job Description")
job_description= st.text_area ("Enter the job description")


# In[9]:

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# In[10]:
# Custom CSS for button styling
st.markdown(
    """
    <style>
        div.stButton > button {
            background-color: #0071ff !important; /* Change button color */
            color: white !important; /* Change text color */
            border-radius: 8px !important; /* Rounded corners */
            padding: 10px 20px !important; /* Adjust padding */
            font-size: 16px !important; /* Change font size */
            font-weight: bold !important; /* Make text bold */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# In[11]:
# Rank Resumes Button
if st.button("Rank Resumes",use_container_width=True):
    if job_description and uploaded_files:
        st.header("Ranking Resumes")

        # Extract text from uploaded resumes
        resumes = [extract_text_from_pdf(file) for file in uploaded_files]

        # Rank resumes based on job description
        scores = rank_resumes(job_description, resumes)

        # Creating a DataFrame to display results
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        st.success("Resumes Ranked Successfully!")
        st.write(results)  # Displaying ranking results

        # Convert results to CSV format
        csv = results.to_csv(index=False)

        # Custom CSS for styling both Rank Resumes and Download buttons
        st.markdown(
            """
            <style>
                /* Styling for all buttons */
                div.stButton > button, div.stDownloadButton > button {
                    background-color: #0071ff !important; /* Blue button */
                    color: white !important; /* White text */
                    border-radius: 8px !important; /* Rounded corners */
                    padding: 10px 20px !important; /* Adjust padding */
                    font-size: 16px !important; /* Bigger text */
                    font-weight: bold !important; /* Bold text */
                    border: none !important; /* Remove border */
                    transition: background-color 0.3s ease !important; /* Smooth hover effect */
                }
                
                /* Hover effect */
                div.stButton > button:hover, div.stDownloadButton > button:hover {
                    background-color: #0057cc !important; /* Darker blue on hover */
                }
            </style>
            """,
            unsafe_allow_html=True
        )


        # Create a download button
        st.download_button(
            label="Download Ranking Results",
            data=csv,
            file_name="resume_ranking_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please provide a job description and upload resumes before ranking.")

# In[11]:
    

# In[ ]:

    



