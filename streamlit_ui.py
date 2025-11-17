import streamlit as st
import sys
import io
from contextlib import contextmanager
import PyPDF2
from dataclasses import dataclass
from typing import List

# Deps from local modules
from recommender import phase1_recommend, phase2_recommend
from infrastructure.database import get_postings_by_ids, Resume
from resume_optimization import run_evolution
from explainability_rag import explain_recommendation

# Page configuration
st.set_page_config(
    page_title="Resume Recommendation System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .job-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        background-color: #f9f9f9;
    }
    .job-title {
        font-size: 18px;
        font-weight: 600;
        color: #1f1f1f;
        margin-bottom: 8px;
    }
    .job-meta {
        font-size: 14px;
        color: #666;
        margin-bottom: 12px;
    }
    .job-description {
        font-size: 14px;
        line-height: 1.5;
        max-height: 100px;
        overflow-y: auto;
        padding: 8px;
        background-color: white;
        border-radius: 4px;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background-color: #4CAF50;
        color: white;
        font-size: 12px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Context manager to capture stdout
@contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout

# Initialize session state
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'phase1_results' not in st.session_state:
    st.session_state.phase1_results = None
if 'phase2_results' not in st.session_state:
    st.session_state.phase2_results = None
if 'selected_postings' not in st.session_state:
    st.session_state.selected_postings = set()
if 'debug_log' not in st.session_state:
    st.session_state.debug_log = ""

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def display_job_posting(posting, score, index, phase):
    """Display a single job posting card with checkbox"""
    col1, col2 = st.columns([0.05, 0.95])
    
    posting_key = f"{phase}_{posting.id}"
    
    with col1:
        is_selected = st.checkbox(
            "Select",
            key=f"check_{posting_key}",
            value=posting_key in st.session_state.selected_postings,
            label_visibility="hidden"
        )
        if is_selected:
            st.session_state.selected_postings.add(posting_key)
        elif posting_key in st.session_state.selected_postings:
            st.session_state.selected_postings.remove(posting_key)
    
    with col2:
        st.markdown(f"""
        <div class="job-card">
            <div class="job-title">
                {posting.title}
                <span class="score-badge">Score: {score:.3f}</span>
            </div>
            <div class="job-meta">
                <strong>{posting.company_name}</strong> • {posting.location or 'Location not specified'}
            </div>
            <div class="job-description">
                {' '.join(posting.description.split()[:100])}...
            </div>
        </div>
        """, unsafe_allow_html=True)

# Header
st.title("📄 Resume Recommendation System")
st.markdown("---")

# Sidebar for resume upload
with st.sidebar:
    st.header("Upload Resume")
    
    upload_method = st.radio(
        "Choose upload method:",
        ["Text Input", "PDF Upload"]
    )
    
    if upload_method == "Text Input":
        resume_text = st.text_area(
            "Paste resume text:",
            height=300,
            placeholder="Paste your resume text here..."
        )
        if st.button("Submit Resume", type="primary"):
            if resume_text:
                st.session_state.resume_text = resume_text
                st.session_state.phase1_results = None
                st.session_state.phase2_results = None
                st.session_state.selected_postings = set()
                st.success("Resume uploaded successfully!")
            else:
                st.error("Please enter resume text")
    
    else:
        uploaded_file = st.file_uploader(
            "Upload PDF:",
            type=['pdf']
        )
        if uploaded_file and st.button("Process PDF", type="primary"):
            with st.spinner("Extracting text from PDF..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                if resume_text:
                    st.session_state.resume_text = resume_text
                    st.session_state.phase1_results = None
                    st.session_state.phase2_results = None
                    st.session_state.selected_postings = set()
                    st.success("PDF processed successfully!")
    
    st.markdown("---")
    
    # Display current resume status
    if st.session_state.resume_text:
        st.success("✓ Resume loaded")
        with st.expander("View Resume Preview"):
            st.text(st.session_state.resume_text[:500] + "...")

# Main content area
if st.session_state.resume_text:
    
    # Recommendation controls
    col1, col2 = st.columns([0.3, 0.7])
    
    with col1:
        st.subheader("Get Recommendations")
        k_value = st.selectbox(
            "Number of recommendations (k):",
            options=[5, 10, 15, 20],
            index=1
        )
        
        if st.button("🔍 Generate Recommendations", type="primary", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                with capture_stdout() as buffer:
                    try:
                        # Create temporary resume object
                        resume = Resume(id=0, text=st.session_state.resume_text)
                        
                        # Phase 1
                        scores1, ids1 = phase1_recommend(resume)
                        st.session_state.phase1_results = {
                            'scores': scores1 if isinstance(scores1, list) else scores1[0].tolist(),
                            'ids': ids1[:k_value]
                        }
                        
                        # Phase 2
                        scores2, ids2 = phase2_recommend(resume, ids1[:k_value])
                        st.session_state.phase2_results = {
                            'scores': list(scores2)[:k_value],
                            'ids': list(ids2)[:k_value]
                        }
                        
                        st.session_state.debug_log += buffer.getvalue()
                        st.success(f"Generated top {k_value} recommendations!")
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
                        st.session_state.debug_log += f"\nError: {str(e)}\n"
    
    with col2:
        st.subheader("Actions")
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🎯 Optimize Resume", use_container_width=True, disabled=len(st.session_state.selected_postings) == 0):
                if st.session_state.selected_postings:
                    with st.spinner("Optimizing resume..."):
                        with capture_stdout() as buffer:
                            try:
                                # Extract posting IDs from selected postings
                                selected_ids = [int(key.split('_')[1]) for key in st.session_state.selected_postings]
                                postings = get_postings_by_ids(selected_ids)
                                
                                result = run_evolution(
                                    base_resume=st.session_state.resume_text,
                                    target_postings=postings
                                )
                                
                                st.session_state.debug_log += buffer.getvalue()
                                
                                # Display results in modal
                                st.session_state.optimization_result = result
                                
                            except Exception as e:
                                st.error(f"Error during optimization: {str(e)}")
                                st.session_state.debug_log += f"\nError: {str(e)}\n"
        
        with action_col2:
            if st.button("💡 Explain Recommendations", use_container_width=True, disabled=len(st.session_state.selected_postings) == 0):
                if st.session_state.selected_postings:
                    with st.spinner("Generating explanations..."):
                        with capture_stdout() as buffer:
                            try:
                                # Extract posting IDs
                                selected_ids = [int(key.split('_')[1]) for key in st.session_state.selected_postings]
                                postings = get_postings_by_ids(selected_ids)
                                
                                # Call explainability function
                                explanation = explain_recommendation(
                                    resume_text=st.session_state.resume_text,
                                    postings=postings
                                )
                                
                                st.session_state.debug_log += buffer.getvalue()
                                st.session_state.explanation_result = explanation
                                
                            except Exception as e:
                                st.error(f"Error generating explanations: {str(e)}")
                                st.session_state.debug_log += f"\nError: {str(e)}\n"
        
        if len(st.session_state.selected_postings) > 0:
            st.info(f"✓ {len(st.session_state.selected_postings)} posting(s) selected")
    
    st.markdown("---")


    
    # Display optimization results if available
    if hasattr(st.session_state, 'optimization_result'):
        with st.expander("📊 Optimization Results", expanded=True):
            result = st.session_state.optimization_result
            st.metric("Best Fitness Score", f"{result['fitness']:.2f}")
            st.text_area("Optimized Resume:", result['resume'], height=400)
            if st.button("Clear Results"):
                delattr(st.session_state, 'optimization_result')
                st.rerun()

    # Display explanation results if available
    if hasattr(st.session_state, 'explanation_result'):
        with st.expander("💡 Match Explanation", expanded=True):
            st.markdown("### Why These Jobs Match Your Resume")
            st.write(st.session_state.explanation_result)
            if st.button("Clear Explanation"):
                delattr(st.session_state, 'explanation_result')
                st.rerun()
    
    # Display recommendations if available
    if st.session_state.phase1_results or st.session_state.phase2_results:
        
        phase_tab1, phase_tab2 = st.tabs(["Phase 1 Recommendations", "Phase 2 Recommendations"])
        
        with phase_tab1:
            if st.session_state.phase1_results:
                st.subheader(f"Top {len(st.session_state.phase1_results['ids'])} Phase 1 Recommendations")
                
                postings = get_postings_by_ids(st.session_state.phase1_results['ids'])
                scores = st.session_state.phase1_results['scores']
                
                for idx, (posting, score) in enumerate(zip(postings, scores)):
                    display_job_posting(posting, score, idx, "phase1")
        
        with phase_tab2:
            if st.session_state.phase2_results:
                st.subheader(f"Top {len(st.session_state.phase2_results['ids'])} Phase 2 Recommendations")
                
                postings = get_postings_by_ids(st.session_state.phase2_results['ids'])
                scores = st.session_state.phase2_results['scores']
                
                for idx, (posting, score) in enumerate(zip(postings, scores)):
                    display_job_posting(posting, score, idx, "phase2")

else:
    # Welcome screen
    st.info("👈 Please upload a resume using the sidebar to get started")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📝 Upload")
        st.write("Submit your resume via text or PDF")
    with col2:
        st.markdown("### 🔍 Discover")
        st.write("Get AI-powered job recommendations")
    with col3:
        st.markdown("### 🎯 Optimize")
        st.write("Tailor your resume for specific roles")

# Debug panel at the bottom
with st.expander("🔧 Debug Panel"):
    st.text_area(
        "System Output:",
        value=st.session_state.debug_log,
        height=300,
        disabled=True
    )
    if st.button("Clear Debug Log"):
        st.session_state.debug_log = ""
        st.rerun()