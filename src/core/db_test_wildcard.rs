
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_search_similar_wildcard_injection() -> Result<()> {
        let temp_dir = tempdir()?;
        let db_path = temp_dir.path().join("test.db");
        let db = Database::new(&db_path)?;

        // Insert a file that should NOT match the search path
        // path: "src/secret.rs"
        let file_path = Path::new("src/secret.rs");
        let file_id = db.insert_file(file_path, "hash1")?;
        
        // Insert a chunk for it
        let chunk_embedding = vec![0.1; 384]; // Dummy embedding
        db.insert_chunk(file_id, 0, "secret content", 1, 10, &chunk_embedding)?;

        // Now search with a path_prefix that contains SQL LIKE wildcards
        // We want to verify if the path prefix is properly escaped.
        // If we search for "src/_", it should match "src/x" but also "src/secret.rs" if _ is treated as wildcard.
        // Wait, "src/_" matching "src/secret.rs"? 
        // "src/_%" in LIKE means: src/ followed by any single character followed by anything.
        // "src/secret.rs" starts with "src/s", so "src/_%" matches "src/s%".
        
        // Let's try a more specific one.
        // File 1: "src/user_data.rs"
        // File 2: "src/userxdata.rs"
        
        // If we search for prefix "src/user_", and it is not escaped, 
        // the query becomes "LIKE 'src/user_%'".
        // This will match "src/user_data.rs" (correct, matches literal _)
        // AND "src/userxdata.rs" (incorrect, matches x via wildcard _)
        
        // Let's setup this scenario.
        
        let path1 = Path::new("src/user_data.rs");
        let id1 = db.insert_file(path1, "h1")?;
        db.insert_chunk(id1, 0, "content1", 1, 10, &chunk_embedding)?;

        let path2 = Path::new("src/userxdata.rs");
        let id2 = db.insert_file(path2, "h2")?;
        db.insert_chunk(id2, 0, "content2", 1, 10, &chunk_embedding)?;

        // Search with prefix "src/user_"
        // We expect it to match ONLY "src/user_data.rs" (and children if any), 
        // but definitely NOT "src/userxdata.rs".
        
        // However, standard `LIKE` treats `_` as single char wildcard.
        // If the code does `format!("{}%", path_prefix_str)`, then "src/user_" becomes "src/user_%".
        // "src/userxdata.rs" matches "src/user_%" because 'x' matches '_'.
        
        let search_path = Path::new("src/user_");
        let results = db.search_similar(&chunk_embedding, search_path, 10)?;
        
        // Check if we got the unwanted file
        let found_unwanted = results.iter().any(|r| r.path == path2.to_path_buf());
        
        if found_unwanted {
            panic!("SQL Injection / Wildcard Leak detected! Search for 'src/user_' matched 'src/userxdata.rs'");
        }

        Ok(())
    }
}
