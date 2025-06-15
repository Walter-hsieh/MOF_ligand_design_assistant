import customtkinter as ctk
from tkinter import filedialog, messagebox
import backend
import threading
import zipfile
import io
import traceback
from PIL import Image

class SmilesToImageWindow(ctk.CTkToplevel):
    # This class is unchanged from your provided version
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("SMILES to Image Converter")
        self.geometry("900x600")
        self.grid_columnconfigure(0, weight=1); self.grid_columnconfigure(1, weight=2); self.grid_rowconfigure(0, weight=1)
        self.results_data = []; self.image_references = []
        self.left_frame = ctk.CTkFrame(self); self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1); self.left_frame.grid_columnconfigure(0, weight=1)
        self.input_label = ctk.CTkLabel(self.left_frame, text="Input Data (Name,SMILES)", font=ctk.CTkFont(size=14, weight="bold")); self.input_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.input_textbox = ctk.CTkTextbox(self.left_frame); self.input_textbox.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.input_textbox.insert("0.0", "Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O\nCaffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        self.generate_button = ctk.CTkButton(self.left_frame, text="Generate Images", command=self.run_generation); self.generate_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        self.right_frame = ctk.CTkFrame(self); self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=1); self.right_frame.grid_columnconfigure(0, weight=1)
        self.results_scroller = ctk.CTkScrollableFrame(self.right_frame, label_text="Generated Images"); self.results_scroller.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.download_button = ctk.CTkButton(self.right_frame, text="Download All as ZIP", command=self.download_zip, state="disabled"); self.download_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
    def run_generation(self):
        input_data = self.input_textbox.get("1.0", "end")
        if not input_data.strip(): return
        self.results_data = backend.process_smiles_to_images(input_data)
        for widget in self.results_scroller.winfo_children(): widget.destroy()
        self.image_references.clear()
        if not self.results_data: return
        valid_image_count = 0
        for result in self.results_data:
            card = ctk.CTkFrame(self.results_scroller, border_width=1, border_color="gray30"); card.pack(pady=5, padx=10, fill="x")
            name_label = ctk.CTkLabel(card, text=result['name'], font=ctk.CTkFont(weight="bold")); name_label.pack(pady=5, padx=10, anchor="w")
            if result['pil_image']:
                valid_image_count += 1
                img_widget = ctk.CTkImage(light_image=result['pil_image'], dark_image=result['pil_image'], size=(200, 200))
                img_label = ctk.CTkLabel(card, image=img_widget, text=""); img_label.pack(pady=5, padx=10)
                self.image_references.append(img_widget)
            else:
                error_label = ctk.CTkLabel(card, text=result['error'], fg_color="#58181F", text_color="white", corner_radius=5, height=50); error_label.pack(pady=20, padx=10, fill="x")
        if valid_image_count > 0: self.download_button.configure(state="normal")
        else: self.download_button.configure(state="disabled")
    def download_zip(self):
        valid_images = [r for r in self.results_data if r['image_bytes']]
        if not valid_images: messagebox.showwarning("No Images", "No valid images to download."); return
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for result in valid_images: zf.writestr(result['filename'], result['image_bytes'])
        zip_buffer.seek(0)
        file_path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("ZIP files", "*.zip")], initialfile="chemical_structures.zip", title="Save ZIP File")
        if file_path:
            with open(file_path, 'wb') as f: f.write(zip_buffer.getvalue())
            messagebox.showinfo("Success", f"Successfully saved to {file_path}")

class ExplanationWindow(ctk.CTkToplevel):
    # This class is unchanged from your provided version
    def __init__(self, ligand_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title(f"Explanation for {ligand_name}")
        self.geometry("600x450")
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=1)
        self.textbox = ctk.CTkTextbox(self, wrap="word", state="disabled", font=("Arial", 14)); self.textbox.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.set_content("Thinking... Please wait for the explanation.")
    def set_content(self, content: str):
        self.textbox.configure(state="normal"); self.textbox.delete("1.0", "end"); self.textbox.insert("1.0", content); self.textbox.configure(state="disabled")

# ==============================================================================
# NEW DATABASE EXPLORER WINDOW CLASS
# ==============================================================================
class DatabaseExplorerWindow(ctk.CTkToplevel):
    """A new window to display extracted MOF data in a grid."""
    def __init__(self, mof_data: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Database Explorer")
        self.geometry("1200x700")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Extracted MOF Data from Documents")
        self.scrollable_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=2) # MOF Name
        self.scrollable_frame.grid_columnconfigure(1, weight=1) # Metals
        self.scrollable_frame.grid_columnconfigure(2, weight=2) # Ligands
        self.scrollable_frame.grid_columnconfigure(3, weight=2) # Property Name
        self.scrollable_frame.grid_columnconfigure(4, weight=2) # Property Value
        
        self.create_header()
        if mof_data:
            self.populate_grid(mof_data)
        else:
            ctk.CTkLabel(self.scrollable_frame, text="No MOF data was extracted from the documents.").pack(pady=20)

    def create_header(self):
        headers = ["MOF Name", "Metals", "Ligands", "Property Name", "Property Value"]
        for i, header in enumerate(headers):
            header_label = ctk.CTkLabel(self.scrollable_frame, text=header, font=ctk.CTkFont(size=14, weight="bold"))
            header_label.grid(row=0, column=i, padx=10, pady=5, sticky="w")
        
        separator = ctk.CTkFrame(self.scrollable_frame, height=2, fg_color="gray50")
        separator.grid(row=1, column=0, columnspan=5, sticky="ew", padx=5, pady=(0, 10))

    def populate_grid(self, data: list):
        for i, mof_entry in enumerate(data):
            name = ctk.CTkLabel(self.scrollable_frame, text=mof_entry.get("mof_name", "N/A"), wraplength=250, justify="left")
            metals = ctk.CTkLabel(self.scrollable_frame, text=mof_entry.get("metals", "N/A"), wraplength=150, justify="left")
            ligands = ctk.CTkLabel(self.scrollable_frame, text=mof_entry.get("ligands", "N/A"), wraplength=250, justify="left")
            prop_name = ctk.CTkLabel(self.scrollable_frame, text=mof_entry.get("property_name", "N/A"), wraplength=200, justify="left")
            prop_val = ctk.CTkLabel(self.scrollable_frame, text=mof_entry.get("property_value", "N/A"), wraplength=200, justify="left")
            
            name.grid(row=i + 2, column=0, padx=10, pady=5, sticky="w")
            metals.grid(row=i + 2, column=1, padx=10, pady=5, sticky="w")
            ligands.grid(row=i + 2, column=2, padx=10, pady=5, sticky="w")
            prop_name.grid(row=i + 2, column=3, padx=10, pady=5, sticky="w")
            prop_val.grid(row=i + 2, column=4, padx=10, pady=5, sticky="w")

# ==============================================================================
# MAIN APP CLASS - MODIFIED FOR THE NEW FEATURE
# ==============================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Chemical Discovery Assistant")
        self.geometry("1200x750")
        ctk.set_appearance_mode("dark")
        self.data_folder_path = ""
        self.approved_ligands = {}
        self.vector_store = None
        self.smiles_window = None
        self.explanation_windows = {}
        self.db_explorer_window = None
        self.mof_database_cache = [] #<-- NEW: Cache for extracted data

        self.grid_columnconfigure(0, weight=1); self.grid_columnconfigure(1, weight=4); self.grid_columnconfigure(2, weight=2); self.grid_rowconfigure(0, weight=1)
        self.left_frame = ctk.CTkFrame(self, width=200, corner_radius=10)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.pack_propagate(False)
        self.left_frame_label = ctk.CTkLabel(self.left_frame, text="Data Control", font=ctk.CTkFont(size=16, weight="bold"))
        self.left_frame_label.pack(pady=10, padx=20)
        self.select_folder_button = ctk.CTkButton(self.left_frame, text="Select Data Folder", command=self.select_folder)
        self.select_folder_button.pack(pady=10, padx=20, fill="x")
        self.folder_path_label = ctk.CTkLabel(self.left_frame, text="No folder selected.", wraplength=180)
        self.folder_path_label.pack(pady=5, padx=10)
        self.index_button = ctk.CTkButton(self.left_frame, text="Index Data", command=self.run_indexing_thread, state="disabled")
        self.index_button.pack(pady=10, padx=20, fill="x")
        self.indexing_status_label = ctk.CTkLabel(self.left_frame, text="")
        self.indexing_status_label.pack(pady=5, padx=10)
        
        # New DB Explorer Button
        self.db_explorer_button = ctk.CTkButton(self.left_frame, text="Database Explorer", command=self.open_db_explorer, state="disabled")
        self.db_explorer_button.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(self.left_frame, text="Utilities", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(30, 10), padx=20)
        self.smiles_button = ctk.CTkButton(self.left_frame, text="SMILES to Image", command=self.open_smiles_utility)
        self.smiles_button.pack(pady=10, padx=20, fill="x")

        # The rest of your UI layout...
        self.center_frame = ctk.CTkFrame(self, corner_radius=10); self.center_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.center_frame.grid_rowconfigure(1, weight=1); self.center_frame.grid_columnconfigure(0, weight=1)
        self.prompt_label = ctk.CTkLabel(self.center_frame, text="Agent Prompt", font=ctk.CTkFont(size=16, weight="bold")); self.prompt_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        self.prompt_input = ctk.CTkTextbox(self.center_frame, height=100); self.prompt_input.grid(row=0, column=0, padx=10, pady=(40, 10), sticky="ew")
        self.prompt_input.insert("1.0", "Modify the ligand of MOF mentioned in the research papers to enhance its CO2 capacity...")
        self.generate_button = ctk.CTkButton(self.center_frame, text="Run Agent", command=self.run_generation_thread, state="disabled"); self.generate_button.grid(row=0, column=0, padx=10, pady=(150, 10), sticky="ew")
        self.results_frame = ctk.CTkScrollableFrame(self.center_frame, label_text="Candidate Ligands"); self.results_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.right_frame = ctk.CTkFrame(self, width=300, corner_radius=10); self.right_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.right_frame.pack_propagate(False); self.right_frame.grid_rowconfigure(1, weight=1); self.right_frame.grid_rowconfigure(3, weight=2); self.right_frame.grid_columnconfigure(0, weight=1)
        self.approved_label = ctk.CTkLabel(self.right_frame, text="Validated Ligands", font=ctk.CTkFont(size=16, weight="bold")); self.approved_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.approved_list_frame = ctk.CTkScrollableFrame(self.right_frame); self.approved_list_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.synthesis_prompt_input = ctk.CTkTextbox(self.right_frame, height=50); self.synthesis_prompt_input.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        self.synthesis_prompt_input.insert("0.0", "Create a synthesis plan...")
        self.synthesis_button = ctk.CTkButton(self.right_frame, text="Generate Synthesis Plan", command=self.run_synthesis_thread, state="disabled"); self.synthesis_button.grid(row=2, column=0, padx=10, pady=(70,10), sticky="ew")
        self.recipe_output = ctk.CTkTextbox(self.right_frame, state="disabled", wrap="word"); self.recipe_output.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

    def open_smiles_utility(self):
        if self.smiles_window is None or not self.smiles_window.winfo_exists():
            self.smiles_window = SmilesToImageWindow(self); self.smiles_window.focus()
        else: self.smiles_window.focus()

    def open_db_explorer(self):
        if self.db_explorer_window is None or not self.db_explorer_window.winfo_exists():
            if self.mof_database_cache:
                self.db_explorer_window = DatabaseExplorerWindow(mof_data=self.mof_database_cache, master=self)
                self.db_explorer_window.focus()
            else:
                messagebox.showinfo("No Data", "No MOF data has been extracted yet. Please re-index your data.")
        else:
            self.db_explorer_window.focus()

    def copy_to_clipboard(self, text_to_copy: str, button: ctk.CTkButton):
        try:
            self.clipboard_clear(); self.clipboard_append(text_to_copy)
            original_text = button.cget("text")
            button.configure(text="Copied!", fg_color="green")
            self.after(1500, lambda: button.configure(text=original_text, fg_color=("#3B8ED0", "#1F6AA5")))
        except Exception as e: messagebox.showerror("Clipboard Error", f"Could not copy text to clipboard: {e}")

    def request_explanation(self, candidate_data: dict):
        ligand_name = candidate_data.get("name", "N/A")
        if ligand_name in self.explanation_windows and self.explanation_windows[ligand_name].winfo_exists():
            self.explanation_windows[ligand_name].focus(); return
        exp_window = ExplanationWindow(ligand_name=ligand_name, master=self)
        self.explanation_windows[ligand_name] = exp_window
        exp_window.focus()
        threading.Thread(target=self.run_explanation_thread, args=(candidate_data, exp_window), daemon=True).start()

    def run_explanation_thread(self, candidate_data: dict, window: ExplanationWindow):
        original_prompt = self.prompt_input.get("1.0", "end-1c")
        explanation_text = backend.get_explanation_for_ligand(original_prompt=original_prompt, ligand_name=candidate_data.get("name"), ligand_smiles=candidate_data.get("smiles"), vector_store_to_use=self.vector_store)
        self.after(0, window.set_content, explanation_text)

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.data_folder_path = path
            self.folder_path_label.configure(text=f"Folder:\n{path}")
            self.index_button.configure(state="normal"); self.generate_button.configure(state="disabled"); self.synthesis_button.configure(state="disabled")
            self.db_explorer_button.configure(state="disabled")
            self.indexing_status_label.configure(text="Folder selected. Ready to index.")
    
    def run_indexing_thread(self):
        self.index_button.configure(state="disabled", text="Indexing...")
        threading.Thread(target=self.index_data, daemon=True).start()

    def index_data(self):
        try:
            self.vector_store, self.mof_database_cache = backend.load_and_index_data(self.data_folder_path)
            if self.vector_store:
                self.indexing_status_label.configure(text="All tasks complete!", text_color="lightgreen")
                self.generate_button.configure(state="normal"); self.synthesis_button.configure(state="normal")
                self.db_explorer_button.configure(state="normal")
            else:
                self.indexing_status_label.configure(text="Indexing Failed.", text_color="orange")
        except Exception as e:
            traceback.print_exc(); self.indexing_status_label.configure(text="CRITICAL ERROR!", text_color="red")
        finally:
            self.index_button.configure(state="normal", text="Re-Index Data")

    def run_generation_thread(self):
        self.generate_button.configure(state="disabled", text="Running Agent...")
        threading.Thread(target=self.generate_ligands, daemon=True).start()

    def generate_ligands(self):
        for widget in self.results_frame.winfo_children(): widget.destroy()
        prompt = self.prompt_input.get("1.0", "end-1c")
        if not prompt or not self.vector_store:
            messagebox.showerror("Error", "Please index data first.")
            self.generate_button.configure(state="normal", text="Run Agent"); return
        candidates = backend.generate_ligands_real(prompt, self.vector_store)
        for candidate in candidates:
            card = self.create_ligand_card(self.results_frame, candidate)
            card.pack(pady=5, padx=10, fill="x")
        self.generate_button.configure(state="normal", text="Run Agent")

    def run_synthesis_thread(self):
        self.synthesis_button.configure(state="disabled", text="Generating...")
        threading.Thread(target=self.generate_synthesis, daemon=True).start()

    def generate_synthesis(self):
        ligand_names = list(self.approved_ligands.keys())
        if not ligand_names:
            messagebox.showwarning("Warning", "Please approve at least one ligand first."); self.synthesis_button.configure(state="normal", text="Generate Synthesis Plan"); return
        prompt = self.synthesis_prompt_input.get("1.0", "end-1c")
        recipe = backend.generate_synthesis_real(prompt, ligand_names, self.vector_store)
        self.recipe_output.configure(state="normal"); self.recipe_output.delete("1.0", "end"); self.recipe_output.insert("0.0", recipe); self.recipe_output.configure(state="disabled")
        self.synthesis_button.configure(state="normal", text="Generate Synthesis Plan")

    def create_ligand_card(self, parent, candidate_data):
        card = ctk.CTkFrame(parent, border_width=1, border_color="gray30")
        name_label = ctk.CTkLabel(card, text=candidate_data.get("name", "N/A"), font=ctk.CTkFont(weight="bold"), wraplength=300)
        name_label.pack(pady=(5,2), padx=10, anchor="w")
        smiles_label = ctk.CTkLabel(card, text=f'SMILES: {candidate_data.get("smiles", "N/A")}', font=ctk.CTkFont(size=10), wraplength=400)
        smiles_label.pack(pady=(0,5), padx=10, anchor="w")
        img_placeholder = ctk.CTkFrame(card, height=100, fg_color="gray20"); img_placeholder.pack(pady=5, padx=10, fill="x")
        img_label = ctk.CTkLabel(img_placeholder, text="Structure Visualization"); img_label.pack(expand=True)
        btn_frame = ctk.CTkFrame(card, fg_color="transparent"); btn_frame.pack(pady=5, padx=10, fill="x")
        btn_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        approve_btn = ctk.CTkButton(btn_frame, text="✅ Approve", fg_color="green", hover_color="darkgreen", command=lambda c=card, d=candidate_data: self.approve_ligand(c, d)); approve_btn.grid(row=0, column=0, padx=2)
        reject_btn = ctk.CTkButton(btn_frame, text="❌ Reject", fg_color="red", hover_color="darkred", command=lambda c=card: c.destroy()); reject_btn.grid(row=0, column=1, padx=2)
        copy_text = f"{candidate_data.get('name', 'N/A')},{candidate_data.get('smiles', 'N/A')}"
        copy_btn = ctk.CTkButton(btn_frame, text="📋 Copy"); copy_btn.configure(command=lambda text=copy_text, btn=copy_btn: self.copy_to_clipboard(text, btn)); copy_btn.grid(row=0, column=2, padx=2)
        explanation_btn = ctk.CTkButton(btn_frame, text="💡 Explain", command=lambda d=candidate_data: self.request_explanation(d)); explanation_btn.grid(row=0, column=3, padx=2)
        return card

    def approve_ligand(self, card_widget, ligand_data):
        name = ligand_data.get("name")
        if name and name not in self.approved_ligands:
            self.approved_ligands[name] = ligand_data.get("smiles")
            self.update_approved_list()
        card_widget.destroy()

    def update_approved_list(self):
        for widget in self.approved_list_frame.winfo_children(): widget.destroy()
        for name in self.approved_ligands.keys():
            label = ctk.CTkLabel(self.approved_list_frame, text=name, wraplength=250); label.pack(pady=2, padx=5, anchor="w")

if __name__ == "__main__":
    app = App()
    app.mainloop()