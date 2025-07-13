import os
import shutil
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pysqlite3 as sqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
load_dotenv()

st.set_page_config(page_title="🤖 Broker-gpt", layout="wide")

# === PAGE ROUTING ===
query_params = st.query_params
page = query_params.get("page", "main")

if page == "manage":
    # === MANAGEMENT PAGE ===
    st.title("🛠️ Broker-gpt Management Dashboard")
    st.markdown("---")
    
    # Initialize directories
    company_base_dir = "data/pdfs"
    vectorstore_base_dir = "vectorstores"
    os.makedirs(company_base_dir, exist_ok=True)
    os.makedirs(vectorstore_base_dir, exist_ok=True)
    
    # Navigation
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Main App"):
            st.query_params.clear()
            st.rerun()
    
    # === SYSTEM OVERVIEW ===
    st.header("📊 System Overview")
    
    # Get system stats
    company_folders = []
    if os.path.exists(company_base_dir):
        company_folders = [f for f in os.listdir(company_base_dir) if os.path.isdir(os.path.join(company_base_dir, f))]
    
    total_pdfs = 0
    total_vectorstores = 0
    
    for company in company_folders:
        company_path = os.path.join(company_base_dir, company)
        if os.path.exists(company_path):
            pdfs = [f for f in os.listdir(company_path) if f.endswith('.pdf')]
            total_pdfs += len(pdfs)
        
        vectorstore_path = os.path.join(vectorstore_base_dir, company)
        if os.path.exists(vectorstore_path):
            total_vectorstores += 1
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Companies", len(company_folders))
    with col2:
        st.metric("Total PDFs", total_pdfs)
    with col3:
        st.metric("Vector Databases", total_vectorstores)
    with col4:
        # Calculate approximate storage
        total_size = 0
        for root, dirs, files in os.walk(company_base_dir):
            for file in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, file))
                except:
                    pass
        for root, dirs, files in os.walk(vectorstore_base_dir):
            for file in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, file))
                except:
                    pass
        st.metric("Storage Used", f"{total_size / (1024*1024):.1f} MB")
    
    st.markdown("---")
    
    # === COMPANY MANAGEMENT ===
    st.header("🏢 Company Management")
    
    if company_folders:
        st.subheader("📋 Existing Companies")
        
        for company in company_folders:
            with st.expander(f"📂 {company}"):
                company_path = os.path.join(company_base_dir, company)
                vectorstore_path = os.path.join(vectorstore_base_dir, company)
                
                # Company stats
                pdfs = []
                if os.path.exists(company_path):
                    pdfs = [f for f in os.listdir(company_path) if f.endswith('.pdf')]
                
                has_vectorstore = os.path.exists(vectorstore_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**PDFs:** {len(pdfs)}")
                    if pdfs:
                        for pdf in pdfs:
                            st.write(f"  • {pdf}")
                with col2:
                    st.write(f"**Vector Database:** {'✅ Yes' if has_vectorstore else '❌ No'}")
                
                # Company actions
                st.markdown("**Actions:**")
                action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                
                with action_col1:
                    if st.button(f"🔄 Relearn", key=f"relearn_{company}"):
                        with st.spinner(f"Relearning {company}..."):
                            try:
                                from ingest import ingest_company_pdfs
                                if os.path.exists(vectorstore_path):
                                    shutil.rmtree(vectorstore_path)
                                ingest_company_pdfs(company)
                                st.success(f"✅ {company} relearned!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                with action_col2:
                    if st.button(f"🗑️ Delete Vector DB", key=f"delete_vectordb_{company}"):
                        try:
                            # Force cleanup of any active Chroma instances
                            import gc
                            gc.collect()
                            
                            if os.path.exists(vectorstore_path):
                                # Try multiple approaches to remove the directory
                                for attempt in range(3):
                                    try:
                                        shutil.rmtree(vectorstore_path)
                                        st.success(f"✅ Deleted vector database for {company}")
                                        st.rerun()
                                        break
                                    except PermissionError:
                                        if attempt < 2:  # Try up to 3 times
                                            import time
                                            time.sleep(0.5)
                                            continue
                                        else:
                                            # If still failing, try to remove files individually
                                            for root, dirs, files in os.walk(vectorstore_path, topdown=False):
                                                for file in files:
                                                    try:
                                                        os.chmod(os.path.join(root, file), 0o777)
                                                        os.remove(os.path.join(root, file))
                                                    except:
                                                        pass
                                                for dir in dirs:
                                                    try:
                                                        os.rmdir(os.path.join(root, dir))
                                                    except:
                                                        pass
                                            try:
                                                os.rmdir(vectorstore_path)
                                                st.success(f"✅ Deleted vector database for {company}")
                                                st.rerun()
                                            except:
                                                st.warning(f"⚠️ Could not fully delete vector database for {company}. Files may be in use.")
                            else:
                                st.info(f"ℹ️ No vector database found for {company}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.info("💡 Try restarting the Streamlit app to release file locks.")
                
                with action_col3:
                    if st.button(f"📄 Delete PDFs", key=f"delete_pdfs_{company}"):
                        try:
                            if os.path.exists(company_path):
                                shutil.rmtree(company_path)
                                os.makedirs(company_path, exist_ok=True)
                            st.success(f"✅ Deleted all PDFs for {company}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                
                with action_col4:
                    if st.button(f"💀 Delete Company", key=f"delete_company_{company}"):
                        try:
                            if os.path.exists(company_path):
                                shutil.rmtree(company_path)
                            if os.path.exists(vectorstore_path):
                                shutil.rmtree(vectorstore_path)
                            st.success(f"✅ Deleted {company} completely")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("No companies found. Create one in the main app first.")
    
    st.markdown("---")
    
    # === BULK OPERATIONS ===
    st.header("🔥 Bulk Operations")
    st.warning("⚠️ **DANGER ZONE** - These operations affect ALL data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧹 Clear All Vector Databases")
        st.write("Removes all learned knowledge but keeps PDF files")
        if st.button("Clear All Vector DBs", type="secondary"):
            try:
                # Force cleanup of any active Chroma instances
                import gc
                gc.collect()
                
                # Try to remove individual company vectorstores
                cleared_count = 0
                failed_companies = []
                
                if os.path.exists(vectorstore_base_dir):
                    for company in os.listdir(vectorstore_base_dir):
                        company_vectorstore = os.path.join(vectorstore_base_dir, company)
                        if os.path.isdir(company_vectorstore):
                            try:
                                # Try multiple approaches to remove the directory
                                for attempt in range(3):
                                    try:
                                        shutil.rmtree(company_vectorstore)
                                        cleared_count += 1
                                        break
                                    except PermissionError:
                                        if attempt < 2:  # Try up to 3 times
                                            import time
                                            time.sleep(0.5)
                                            continue
                                        else:
                                            # If still failing, try to remove files individually
                                            for root, dirs, files in os.walk(company_vectorstore, topdown=False):
                                                for file in files:
                                                    try:
                                                        os.chmod(os.path.join(root, file), 0o777)
                                                        os.remove(os.path.join(root, file))
                                                    except:
                                                        pass
                                                for dir in dirs:
                                                    try:
                                                        os.rmdir(os.path.join(root, dir))
                                                    except:
                                                        pass
                                            try:
                                                os.rmdir(company_vectorstore)
                                                cleared_count += 1
                                            except:
                                                failed_companies.append(company)
                            except Exception as e:
                                failed_companies.append(company)
                
                # Recreate the vectorstores directory
                os.makedirs(vectorstore_base_dir, exist_ok=True)
                
                if cleared_count > 0:
                    st.success(f"✅ Cleared {cleared_count} vector database(s)!")
                
                if failed_companies:
                    st.warning(f"⚠️ Could not clear: {', '.join(failed_companies)}. Files may be in use. Try stopping and restarting the app.")
                
                if cleared_count == 0 and not failed_companies:
                    st.info("ℹ️ No vector databases found to clear.")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try restarting the Streamlit app to release file locks.")
    
    with col2:
        st.subheader("💣 Nuclear Reset")
        st.write("Deletes EVERYTHING - all PDFs, databases, and companies")
        
        if st.button("🚨 NUCLEAR RESET", type="primary"):
            st.session_state["confirm_nuclear"] = True
        
        if st.session_state.get("confirm_nuclear", False):
            st.error("⚠️ This will DELETE EVERYTHING! Are you sure?")
            confirm_col1, confirm_col2 = st.columns(2)
            
            with confirm_col1:
                if st.button("❌ Cancel Nuclear"):
                    st.session_state["confirm_nuclear"] = False
                    st.rerun()
            
            with confirm_col2:
                if st.button("☢️ YES, NUKE IT ALL"):
                    try:
                        # Force cleanup of any active instances
                        import gc
                        gc.collect()
                        
                        # Nuclear deletion with retry logic
                        deleted_companies = []
                        failed_companies = []
                        
                        # Try to delete everything
                        for attempt in range(3):
                            try:
                                if os.path.exists(company_base_dir):
                                    shutil.rmtree(company_base_dir)
                                if os.path.exists(vectorstore_base_dir):
                                    shutil.rmtree(vectorstore_base_dir)
                                break
                            except PermissionError:
                                if attempt < 2:
                                    import time
                                    time.sleep(1)
                                    continue
                                else:
                                    # Manual file-by-file deletion
                                    for base_dir in [company_base_dir, vectorstore_base_dir]:
                                        if os.path.exists(base_dir):
                                            for root, dirs, files in os.walk(base_dir, topdown=False):
                                                for file in files:
                                                    try:
                                                        file_path = os.path.join(root, file)
                                                        os.chmod(file_path, 0o777)
                                                        os.remove(file_path)
                                                    except:
                                                        pass
                                                for dir in dirs:
                                                    try:
                                                        os.rmdir(os.path.join(root, dir))
                                                    except:
                                                        pass
                                            try:
                                                os.rmdir(base_dir)
                                            except:
                                                pass
                        
                        # Recreate directories
                        os.makedirs(company_base_dir, exist_ok=True)
                        os.makedirs(vectorstore_base_dir, exist_ok=True)
                        
                        # Clear session state
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        
                        st.success("☢️ NUCLEAR RESET COMPLETE! All data destroyed.")
                        st.balloons()
                        st.info("💡 If some files couldn't be deleted, restart the app to fully complete the reset.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Nuclear reset failed: {str(e)}")
                        st.info("💡 Try restarting the Streamlit app to release file locks.")
                        st.session_state["confirm_nuclear"] = False

else:
    # === MAIN APPLICATION PAGE ===
    st.title("🤝 Broker-gpt: Insurance Broker Assistant")
    
    # Add management link
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🛠️ Management Dashboard"):
            st.query_params["page"] = "manage"
            st.rerun()
    with col2:
        st.write("Access admin functions, bulk operations, and system overview")
    st.markdown("---")
    
    # === Sidebar UI ===
    st.sidebar.header("🏢 Manage Companies")
    company_base_dir = "data/pdfs"
    vectorstore_base_dir = "vectorstores"
    os.makedirs(company_base_dir, exist_ok=True)
    os.makedirs(vectorstore_base_dir, exist_ok=True)
    
    # Initialize session state for company refresh
    if "company_added" not in st.session_state:
        st.session_state["company_added"] = False
    
    # Add new company
    new_company = st.sidebar.text_input("➕ Create new company folder", key="new_company")
    if st.sidebar.button("Add Company"):
        new_path = os.path.join(company_base_dir, new_company)
        if new_company and not os.path.exists(new_path):
            os.makedirs(new_path)
            st.sidebar.success(f"✅ Added company: {new_company}")
            st.session_state["company_added"] = True
        else:
            st.sidebar.warning("⚠️ Folder exists or name is empty")
    
    # Refresh company list - check if directory exists first
    company_folders = []
    if os.path.exists(company_base_dir):
        company_folders = [f for f in os.listdir(company_base_dir) if os.path.isdir(os.path.join(company_base_dir, f))]
    
    if not company_folders:
        st.warning("⚠️ No companies found. Add one to begin.")
        st.stop()
    
    # Select company
    selected_company = st.sidebar.radio("📂 Select company", company_folders, key="selected_company")
    
    # Upload PDF
    uploaded_pdf = st.sidebar.file_uploader("📄 Upload PDF to selected company", type="pdf", key="uploader")
    if uploaded_pdf:
        save_path = os.path.join(company_base_dir, selected_company, uploaded_pdf.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.sidebar.success(f"✅ Uploaded: {uploaded_pdf.name}")
    
    # Relearn PDFs
    if st.sidebar.button("🔄 Relearn PDFs"):
        from ingest import ingest_company_pdfs
        shutil.rmtree(os.path.join("vectorstores", selected_company), ignore_errors=True)
        ingest_company_pdfs(selected_company)
        st.sidebar.success("✅ Re-ingested knowledge for " + selected_company)
    
    # === Main Area ===
    vectorstore_path = os.path.join("vectorstores", selected_company)
    if not os.path.exists(vectorstore_path):
        st.info(f"Upload PDFs for **{selected_company}** and click 'Relearn PDFs' to start.")
    else:
        st.markdown("---")
        st.subheader(f"💬 Ask {selected_company} anything about their policies")
        query = st.text_input("Type your question:")
    
        if query:
            with st.spinner("🔍 Please wait while Broker-gpt searches and thinks..."):
                retriever = Chroma(
                    persist_directory=vectorstore_path,
                    embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                ).as_retriever()
    
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs])
    
                GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"""Answer the following question using the context below.
    
    Question: {query}
    
    Context: {context}
    """
                        }]
                    }]
                }
    
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, headers=headers, data=json.dumps(payload))
    
            st.markdown("---")
            st.markdown("### 🤖 Broker-gpt's Answer")
            if response.status_code == 200:
                try:
                    answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                    st.success(answer)
                except:
                    st.error("❌ Gemini replied but parsing failed.")
            else:
                st.error(f"❌ Gemini API Error: {response.status_code}")
                st.json(response.json())
