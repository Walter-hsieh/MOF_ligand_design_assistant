import customtkinter as ctk
from tkinter import filedialog, messagebox
import backend  # Import our backend logic
import threading

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Chemical Discovery Assistant")
        self.geometry("1200x750")
        ctk.set_appearance_mode("dark")

        # --- DATA & STATE ---
        self.data_folder_path = ""
        self.approved_ligands = {}
        self.vector_store = None # Will hold the indexed data

        # --- LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_columnconfigure(2, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL: DATA & HISTORY ---
        self.left_frame = ctk.CTkFrame(self, width=200, corner_radius=10)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_propagate(False)
        
        self.left_frame_label = ctk.CTkLabel(self.left_frame, text="Data Control", font=ctk.CTkFont(size=16, weight="bold"))
        self.left_frame_label.pack(pady=10)

        self.select_folder_button = ctk.CTkButton(self.left_frame, text="Select Data Folder", command=self.select_folder)
        self.select_folder_button.pack(pady=10, padx=20, fill="x")

        self.folder_path_label = ctk.CTkLabel(self.left_frame, text="No folder selected.", wraplength=180)
        self.folder_path_label.pack(pady=5, padx=10)

        self.index_button = ctk.CTkButton(self.left_frame, text="Index Data", command=self.run_indexing_thread, state="disabled")
        self.index_button.pack(pady=10, padx=20, fill="x")
        
        self.indexing_status_label = ctk.CTkLabel(self.left_frame, text="")
        self.indexing_status_label.pack(pady=5, padx=10)

        # --- CENTER PANEL: MAIN WORKSPACE ---
        self.center_frame = ctk.CTkFrame(self, corner_radius=10)
        self.center_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.center_frame.grid_rowconfigure(1, weight=1)
        self.center_frame.grid_columnconfigure(0, weight=1)

        self.prompt_label = ctk.CTkLabel(self.center_frame, text="Input 1: Ligand Design Prompt", font=ctk.CTkFont(size=16, weight="bold"))
        self.prompt_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.prompt_input = ctk.CTkTextbox(self.center_frame, height=100)
        self.prompt_input.grid(row=0, column=0, padx=10, pady=(40, 10), sticky="ew")

        self.generate_button = ctk.CTkButton(self.center_frame, text="Generate Candidates", command=self.run_generation_thread, state="disabled")
        self.generate_button.grid(row=0, column=0, padx=10, pady=(150, 10), sticky="ew")
        
        self.results_frame = ctk.CTkScrollableFrame(self.center_frame, label_text="Candidate Ligands")
        self.results_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # --- RIGHT PANEL: SYNTHESIS ---
        self.right_frame = ctk.CTkFrame(self, width=300, corner_radius=10)
        self.right_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_propagate(False)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_rowconfigure(3, weight=2)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.approved_label = ctk.CTkLabel(self.right_frame, text="Validated Ligands", font=ctk.CTkFont(size=16, weight="bold"))
        self.approved_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.approved_list_frame = ctk.CTkScrollableFrame(self.right_frame)
        self.approved_list_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        self.synthesis_prompt_input = ctk.CTkTextbox(self.right_frame, height=50)
        self.synthesis_prompt_input.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        self.synthesis_prompt_input.insert("0.0", "Create a synthesis plan...")

        self.synthesis_button = ctk.CTkButton(self.right_frame, text="Generate Synthesis Plan", command=self.run_synthesis_thread, state="disabled")
        self.synthesis_button.grid(row=2, column=0, padx=10, pady=(70,10), sticky="ew")

        self.recipe_output = ctk.CTkTextbox(self.right_frame, state="disabled", wrap="word")
        self.recipe_output.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        
    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.data_folder_path = path
            self.folder_path_label.configure(text=f"Folder:\n{path}")
            self.index_button.configure(state="normal")
            self.generate_button.configure(state="disabled")
            self.indexing_status_label.configure(text="Folder selected. Ready to index.")
    
    def run_indexing_thread(self):
        self.index_button.configure(state="disabled", text="Indexing...")
        # Run the backend function in a separate thread to keep the UI responsive
        threading.Thread(target=self.index_data, daemon=True).start()

    def index_data(self):
        self.vector_store = backend.load_and_index_data(self.data_folder_path)
        if self.vector_store:
            self.indexing_status_label.configure(text="Indexing Complete!", text_color="lightgreen")
            self.generate_button.configure(state="normal") # Enable generation after indexing
            self.synthesis_button.configure(state="normal")
        else:
            self.indexing_status_label.configure(text="Indexing Failed.", text_color="red")
        self.index_button.configure(state="normal", text="Re-Index Data")

    def run_generation_thread(self):
        self.generate_button.configure(state="disabled", text="Generating...")
        threading.Thread(target=self.generate_ligands, daemon=True).start()

    def generate_ligands(self):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        prompt = self.prompt_input.get("1.0", "end-1c")
        if not prompt or not self.vector_store:
            messagebox.showerror("Error", "Please index data and enter a prompt first.")
            self.generate_button.configure(state="normal", text="Generate Candidates")
            return

        candidates = backend.generate_ligands_real(prompt, self.vector_store)
        
        # Populate results
        for i, candidate in enumerate(candidates):
            card = self.create_ligand_card(self.results_frame, candidate)
            card.pack(pady=5, padx=10, fill="x")
        
        self.generate_button.configure(state="normal", text="Generate Candidates")

    def run_synthesis_thread(self):
        self.synthesis_button.configure(state="disabled", text="Generating...")
        threading.Thread(target=self.generate_synthesis, daemon=True).start()

    def generate_synthesis(self):
        ligand_names = list(self.approved_ligands.keys())
        if not ligand_names:
            messagebox.showwarning("Warning", "Please approve at least one ligand first.")
            self.synthesis_button.configure(state="normal", text="Generate Synthesis Plan")
            return
            
        prompt = self.synthesis_prompt_input.get("1.0", "end-1c")
        recipe = backend.generate_synthesis_real(prompt, ligand_names, self.vector_store)
        
        self.recipe_output.configure(state="normal")
        self.recipe_output.delete("1.0", "end")
        self.recipe_output.insert("0.0", recipe)
        self.recipe_output.configure(state="disabled")
        self.synthesis_button.configure(state="normal", text="Generate Synthesis Plan")

    def create_ligand_card(self, parent, candidate_data):
        card = ctk.CTkFrame(parent, border_width=1, border_color="gray30")
        
        name_label = ctk.CTkLabel(card, text=candidate_data.get("name", "N/A"), font=ctk.CTkFont(weight="bold"), wraplength=300)
        name_label.pack(pady=(5,2), padx=10, anchor="w")
        
        smiles_label = ctk.CTkLabel(card, text=f'SMILES: {candidate_data.get("smiles", "N/A")}', font=ctk.CTkFont(size=10), wraplength=400)
        smiles_label.pack(pady=(0,5), padx=10, anchor="w")
        
        img_placeholder = ctk.CTkFrame(card, height=100, fg_color="gray20")
        img_placeholder.pack(pady=5, padx=10, fill="x")
        img_label = ctk.CTkLabel(img_placeholder, text="Structure Visualization")
        img_label.pack(expand=True)
        
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.pack(pady=5, padx=10, fill="x")
        btn_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        approve_btn = ctk.CTkButton(btn_frame, text="✅ Approve", fg_color="green", hover_color="darkgreen",
                                    command=lambda c=card, d=candidate_data: self.approve_ligand(c, d))
        approve_btn.grid(row=0, column=0, padx=5)
        
        reject_btn = ctk.CTkButton(btn_frame, text="❌ Reject", fg_color="red", hover_color="darkred",
                                   command=lambda c=card: c.destroy())
        reject_btn.grid(row=0, column=1, padx=5)
        
        feedback_btn = ctk.CTkButton(btn_frame, text="💬 Feedback", 
                                     command=lambda name=candidate_data.get("name"): self.get_feedback(name))
        feedback_btn.grid(row=0, column=2, padx=5)
        
        return card
        
    def approve_ligand(self, card_widget, ligand_data):
        name = ligand_data.get("name")
        if name and name not in self.approved_ligands:
            self.approved_ligands[name] = ligand_data.get("smiles")
            self.update_approved_list()
        card_widget.destroy()
    
    def update_approved_list(self):
        for widget in self.approved_list_frame.winfo_children():
            widget.destroy()
        for name in self.approved_ligands.keys():
            label = ctk.CTkLabel(self.approved_list_frame, text=name, wraplength=250)
            label.pack(pady=2, padx=5, anchor="w")

    def get_feedback(self, ligand_name):
        dialog = ctk.CTkInputDialog(text=f"Enter feedback for {ligand_name}:", title="Provide Feedback")
        feedback = dialog.get_input()
        if feedback:
            print(f"Feedback received for {ligand_name}: {feedback}")

if __name__ == "__main__":
    app = App()
    app.mainloop()