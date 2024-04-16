import streamlit as st   # for making website
import pickle  # for loading file
import re  # for cleaning the data(resume)
import nltk

nltk.download('punkt')         # used in backend
nltk.download('stopwords')

# loading the models clf and tfidf
clf = pickle.load(open('.\clf.pkl','rb'))
tfidfd = pickle.load(open('.\tfidf.pkl','rb'))

def CleanResume(txt):
  cleanTxt = re.sub('http\S+\s',' ',txt)
  cleanTxt = re.sub('RT|cc',' ',cleanTxt)
  cleanTxt = re.sub('#\S+\s',' ',cleanTxt)
  cleanTxt = re.sub('@\S+',' ',cleanTxt)
  cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~"""),' ',cleanTxt)
  cleanTxt = re.sub(r'[^\x00-\x7f]',' ',cleanTxt)
  cleanTxt = re.sub('\s+',' ',cleanTxt)
  return cleanTxt

# web app
# def main():
#   st.title("Resume Screening App")
#   uploaded_file = st.file_uploader("Upload Resume", type=['txt','pdf'])

#   if uploaded_file is not None:
#     try:
#       resume_bytes = uploaded_file.read()
#       resume_text = resume_bytes.decode('utf-8')
#     except UnicodeDecodeError:
#       # if utf-8 decoding fails, try decoding with 'latin-1'
#       resume_text = resume_bytes.decode('latin-1')

#     cleaned_resume = CleanResume(resume_text)
#     input_features = tfidfd.transform([cleaned_resume])
#     prediction_id = clf.predict(cleaned_resume)[0]
#     st.write(prediction_id)

# uploaded_file = st.file_uploader("Upload Resume", type=['txt','pdf'])
# resume_bytes = uploaded_file.read()
# resume_text = resume_bytes.decode('utf-8')
# st.write(resume_text)
# python main
# if __name__ == "__main__":
#   main()

# import streamlit as st
from PyPDF2 import PdfReader

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# File uploader
uploaded_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])

# If file is uploaded
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    # If uploaded file is text
    if file_extension == 'txt':
        resume_text = uploaded_file.getvalue().decode("utf-8")
    # If uploaded file is PDF
    elif file_extension == 'pdf':
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a text file (.txt) or a PDF file (.pdf).")
    
    # Display resume text
    # st.write(resume_text)

    # cleaned_resume = CleanResume(resume_text)
    # input_features = tfidfd.transform([cleaned_resume])
    # prediction_id = clf.predict(cleaned_resume)[0]
    # st.write(prediction_id)
    # st.write(cleaned_resume)

    # Assuming you've already imported necessary libraries and defined `CleanResume` function, `tfidfd`, and `clf`

    # Preprocess the resume text
    cleaned_resume = CleanResume(resume_text)

    # Transform the preprocessed text using tfidfd
    input_features = tfidfd.transform([cleaned_resume])

    # Make a prediction using the classifier
    prediction_id = clf.predict(input_features)[0]

    # Display the prediction

    category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

    category_name = category_mapping.get(prediction_id, "Unknown")

    st.write("Predicted Category:", category_name)

