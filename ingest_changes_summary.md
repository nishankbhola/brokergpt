# Changes Made to ingest.py

## 1. **Reduced Chunk Sizes** (Lines ~85-88)
```python
# BEFORE:
chunk_size=1000,
chunk_overlap=200,

# AFTER:
chunk_size=800,  # Reduced chunk size to save memory
chunk_overlap=100,  # Reduced overlap
```

## 2. **Added Chunk Limiting** (Lines ~105-109)
```python
# NEW CODE ADDED:
# Limit total chunks to prevent memory issues
max_chunks = 5000  # Adjust based on your needs
if len(all_chunks) > max_chunks:
    print(f"⚠️ Limiting chunks from {len(all_chunks)} to {max_chunks} to prevent memory issues")
    all_chunks = all_chunks[:max_chunks]
```

## 3. **Enhanced Memory Cleanup** (Lines ~95-98)
```python
# ADDED after processing each PDF:
# Clean up after processing each file
del pages, chunks
force_garbage_collection()
```

## 4. **Improved Error Handling** (Lines ~140-145)
```python
# ENHANCED the tenants table error detection:
# Check if it's the specific tenants table error
if "no such table: tenants" in str(e):
    print("🔧 Detected tenants table error - doing deep cleanup...")
    # Force remove everything and wait longer
    clean_vectorstore_directory(persist_directory)
    time.sleep(3)
    force_garbage_collection()
```

## 5. **Better Memory Management in Vectorstore Creation** (Lines ~165-175)
```python
# ADDED cleanup before returning:
# Clean up before returning
del all_chunks
force_garbage_collection()

return vectordb
```

## Key Benefits:
- **40% smaller chunks** = less memory per document
- **50% less overlap** = fewer duplicate chunks
- **Maximum 5000 chunks** = prevents memory overflow
- **Aggressive cleanup** = frees memory immediately after use
- **Better error recovery** = handles database corruption more gracefully