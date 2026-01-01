"""
Notes management for AGI-o1.

Provides functions for saving, editing, viewing, and searching notes
stored in docs/paper.txt.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from amarillo.core.constants import NOTES_MARKER, NOTE_HEADING_PREFIX, DEFAULT_PAGE_SIZE, DEFAULT_PAGE


class NotesManager:
    """
    Manages notes stored in a markdown file.
    
    Notes are organized under a special marker section with category/filename keys.
    """
    
    def __init__(self, paper_path: Path):
        """
        Initialize the notes manager.
        
        Args:
            paper_path: Path to the paper.txt file
        """
        self.paper_path = paper_path
        self._ensure_notes_section()
    
    def _ensure_notes_section(self) -> None:
        """Ensure the notes section exists in the paper file."""
        self.paper_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.paper_path.exists():
            self.paper_path.write_text("")
        
        text = self.paper_path.read_text()
        if NOTES_MARKER not in text:
            with self.paper_path.open("a") as handle:
                handle.write("\n\n" + NOTES_MARKER + "\n\n")
    
    def _load_notes(self) -> Tuple[str, Dict[str, str], List[str]]:
        """
        Load notes from the paper file.
        
        Returns:
            Tuple of (before_section, notes_map, order)
        """
        self._ensure_notes_section()
        text = self.paper_path.read_text()
        
        marker_index = text.find(NOTES_MARKER)
        before = text[:marker_index].rstrip("\n")
        after = text[marker_index + len(NOTES_MARKER):]
        
        lines = after.splitlines()
        notes: Dict[str, str] = {}
        order: List[str] = []
        current_key: Optional[str] = None
        current_lines: List[str] = []
        
        for line in lines:
            if line.startswith(NOTE_HEADING_PREFIX):
                if current_key is not None:
                    notes[current_key] = "\n".join(current_lines).strip()
                    order.append(current_key)
                current_key = line[len(NOTE_HEADING_PREFIX):].strip()
                current_lines = []
            else:
                if current_key is not None:
                    current_lines.append(line)
        
        if current_key is not None:
            notes[current_key] = "\n".join(current_lines).strip()
            order.append(current_key)
        
        return before, notes, order
    
    def _write_notes(
        self,
        before_section: str,
        notes_map: Dict[str, str],
        order: List[str]
    ) -> None:
        """
        Write notes back to the paper file.
        
        Args:
            before_section: Content before the notes marker
            notes_map: Dictionary of note keys to content
            order: List of keys in display order
        """
        lines = []
        if before_section:
            lines.append(before_section.rstrip("\n"))
        lines.append(NOTES_MARKER)
        
        for key in order:
            if key not in notes_map:
                continue
            lines.append("")
            lines.append(f"{NOTE_HEADING_PREFIX}{key}")
            content = notes_map[key]
            if content:
                lines.append(content)
        
        final_text = "\n".join(lines)
        if not final_text.endswith("\n"):
            final_text += "\n"
        
        self.paper_path.write_text(final_text)
    
    def save(self, category: str, filename: str, content: str) -> None:
        """
        Save a note to the repository.
        
        Args:
            category: Category for the note
            filename: Filename within the category
            content: Note content
        """
        key = f"{category}/{filename}"
        before, notes_map, order = self._load_notes()
        
        if key not in order:
            order.append(key)
        notes_map[key] = content.strip()
        
        self._write_notes(before, notes_map, order)
        logging.info("Saved paper note: %s", key)
    
    def edit(self, category: str, filename: str, new_content: str) -> str:
        """
        Edit an existing note or create if missing.
        
        Args:
            category: Category of the note
            filename: Filename within the category
            new_content: New content for the note
        
        Returns:
            Status message
        """
        key = f"{category}/{filename}"
        before, notes_map, order = self._load_notes()
        
        created = key not in notes_map
        if created:
            order.append(key)
        notes_map[key] = new_content.strip()
        
        self._write_notes(before, notes_map, order)
        logging.info("Edited paper note: %s (created=%s)", key, created)
        
        if created:
            return f"Created and saved new note '{key}'."
        return f"Updated the content of '{key}'."
    
    def list(
        self,
        category: Optional[str] = None,
        page: int = DEFAULT_PAGE,
        page_size: int = DEFAULT_PAGE_SIZE
    ) -> str:
        """
        List notes, optionally filtered by category.
        
        Args:
            category: Optional category filter
            page: Page number (1-indexed)
            page_size: Number of items per page
        
        Returns:
            Formatted list of notes
        """
        _, notes_map, order = self._load_notes()
        
        files = []
        for key in order:
            if "/" in key:
                cat, _ = key.split("/", 1)
            else:
                cat = ""
            if category and cat != category:
                continue
            files.append(key)
        
        if not files:
            if category:
                return f"No notes found for category '{category}'."
            return "No notes stored in docs/paper.txt."
        
        total_files = len(files)
        total_pages = (total_files + page_size - 1) // page_size
        
        if page < 1 or page > total_pages:
            return f"Invalid page number. There are {total_pages} pages available."
        
        start = (page - 1) * page_size
        end = start + page_size
        paginated_files = files[start:end]
        
        file_list = "\n".join(paginated_files)
        logging.info("Listed paper notes for page %s.", page)
        return f"Available notes (Page {page}/{total_pages}):\n{file_list}"
    
    def view(self, category: str, filename: str) -> str:
        """
        View the content of a note.
        
        Args:
            category: Category of the note
            filename: Filename within the category
        
        Returns:
            Note content or error message
        """
        key = f"{category}/{filename}"
        _, notes_map, _ = self._load_notes()
        
        if key not in notes_map:
            logging.warning("Requested note %s does not exist.", key)
            return f"Note '{key}' does not exist."
        
        content = notes_map[key]
        logging.info("Viewed paper note: %s", key)
        return f"Content of '{key}':\n\n{content}"
    
    def delete(self, category: str, filename: str) -> str:
        """
        Delete a note from the repository.
        
        Args:
            category: Category of the note
            filename: Filename within the category
        
        Returns:
            Status message
        """
        key = f"{category}/{filename}"
        before, notes_map, order = self._load_notes()
        
        if key not in notes_map:
            logging.warning("Attempted to delete missing note: %s", key)
            return f"Note '{key}' does not exist."
        
        notes_map.pop(key)
        order = [k for k in order if k != key]
        
        self._write_notes(before, notes_map, order)
        logging.info("Deleted paper note: %s", key)
        return f"Deleted note '{key}'."
    
    def search(self, query: str) -> str:
        """
        Search for a query string within all notes.
        
        Args:
            query: Search query string
        
        Returns:
            List of matching notes or message if none found
        """
        _, notes_map, order = self._load_notes()
        
        matching = []
        q_lower = query.lower()
        
        for key in order:
            content = notes_map.get(key, "")
            if q_lower in content.lower() or q_lower in key.lower():
                matching.append(key)
        
        if not matching:
            return f"No notes contain the query '{query}'."
        
        file_list = "\n".join(matching)
        logging.info("Search notes for query '%s' -> %d matches.", query, len(matching))
        return f"Notes containing '{query}':\n{file_list}"
    
    def read_all(self) -> str:
        """
        Read all notes and format for display.
        
        Returns:
            Formatted string of all notes with categories
        """
        _, notes_map, order = self._load_notes()
        
        all_content = []
        for key in order:
            if "/" in key:
                category, filename = key.split("/", 1)
            else:
                category, filename = "", key
            content = notes_map.get(key, "")
            all_content.append(f"Category: {category}, File: {filename}\n{content}\n")
        
        return "\n".join(all_content)
