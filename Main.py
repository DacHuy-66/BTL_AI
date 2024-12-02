import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
import io
import threading

import cv2
from PIL import Image, ImageTk

from BTL_AI.ThuatToan_LBPH.Recognize import get_face_recognizer
from BTL_AI.ThuatToan_LBPH.Train import train_face_recognizer


class StudentManagementSystem:
    def __init__(self, master):
        self.master = master
        self.master.title("Hệ thống Điểm Danh Sinh Viên")
        self.master.geometry("1000x750")

        self.conn = sqlite3.connect('students.db')
        self.cursor = self.conn.cursor()
        self.create_tables()

        self.create_widgets()
        self.load_students()

        # Khởi tạo camera
        self.cap = cv2.VideoCapture(0)
        self.is_capturing = False

        # Khởi tạo face recognizer
        self.face_recognizer = get_face_recognizer(self.cursor)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        self.conn.close()
        self.master.destroy()

    def update_camera(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Lật hình ảnh ngang
                recognized_frame, recognized_faces = self.face_recognizer.recognize_face(frame)

                # Vẽ tên và độ tin cậy lên khung hình
                for (x, y, w, h), (name, confidence) in recognized_faces:
                    # Vẽ hình chữ nhật xung quanh khuôn mặt
                    cv2.rectangle(recognized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Chuẩn bị text để hiển thị
                    text = f"{name} ({confidence:.2f}%)"

                    # Tính toán vị trí để đặt text
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = x + (w - text_size[0]) // 2
                    text_y = y - 10 if y - 10 > 10 else y + h + 20

                    # Vẽ nền cho text
                    cv2.rectangle(recognized_frame, (text_x - 5, text_y - text_size[1] - 5),
                                  (text_x + text_size[0] + 5, text_y + 5), (0, 255, 0), cv2.FILLED)

                    # Vẽ text lên khung hình
                    cv2.putText(recognized_frame, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Chuyển đổi frame để hiển thị trong Tkinter
                rgb_frame = cv2.cvtColor(recognized_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)

                # Resize image to fit the label
                img = img.resize((400, 300), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)

                self.camera_label.after(10, self.update_camera)
            else:
                self.is_capturing = False
                self.camera_label.config(image='')

    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS students
                              (msv TEXT PRIMARY KEY,
                               name TEXT,
                               birthdate TEXT,
                               class TEXT)''')

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS face_images
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               msv TEXT,
                               image_number INTEGER,
                               image BLOB,
                               FOREIGN KEY(msv) REFERENCES students(msv))''')
        self.conn.commit()

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ('MSV', 'Tên', 'Ngày sinh', 'Lớp')
        self.tree = ttk.Treeview(left_frame, columns=columns, show='headings')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        details_frame = ttk.LabelFrame(right_frame, text="Thông tin sinh viên")
        details_frame.pack(fill=tk.X, pady=10)

        ttk.Label(details_frame, text="MSV:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.msv_entry = ttk.Entry(details_frame)
        self.msv_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

        ttk.Label(details_frame, text="Tên:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.name_entry = ttk.Entry(details_frame)
        self.name_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

        ttk.Label(details_frame, text="Ngày sinh:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.birthdate_entry = ttk.Entry(details_frame)
        self.birthdate_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

        ttk.Label(details_frame, text="Lớp:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.class_entry = ttk.Entry(details_frame)
        self.class_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Thêm/Cập nhật", command=self.add_or_update_student).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Xóa", command=self.delete_student).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Xem ảnh", command=self.view_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Thêm ảnh", command=self.add_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Train Model", command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Check", command=self.check_face).pack(side=tk.LEFT, padx=5)


        search_frame = ttk.Frame(right_frame)
        search_frame.pack(fill=tk.X, pady=10)

        ttk.Label(search_frame, text="Tìm kiếm:").pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(search_frame, text="Tìm", command=self.search_student).pack(side=tk.LEFT, padx=5)
        # Thêm label để hiển thị hình ảnh từ camera
        self.camera_label = ttk.Label(right_frame)
        self.camera_label.pack(pady=10)

        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)

    def check_face(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.update_camera()
        else:
            self.is_capturing = False
            self.camera_label.config(image='')

    def update_camera(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Lật hình ảnh ngang
                recognized_frame, recognized_faces = self.face_recognizer.recognize_face(frame)

                # Chuyển đổi frame để hiển thị trong Tkinter
                rgb_frame = cv2.cvtColor(recognized_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)

                # Hiển thị thông tin nhận diện (nếu cần)
                if recognized_faces:
                    print(recognized_faces)  # Hoặc hiển thị trong GUI

                self.camera_label.after(10, self.update_camera)
            else:
                self.is_capturing = False
                self.camera_label.config(image='')

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

    def train_model(self):
        self.cursor.execute("SELECT COUNT(*) FROM students")
        student_count = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM face_images")
        image_count = self.cursor.fetchone()[0]

        if student_count == 0 or image_count == 0:
            messagebox.showerror("Lỗi", "Không có đủ dữ liệu để train. Hãy thêm sinh viên và ảnh trước.")
            return

        self.cursor.execute("SELECT msv, image FROM face_images")
        image_data = self.cursor.fetchall()

        threading.Thread(target=self._train_model_thread, args=(image_data,)).start()

    def _train_model_thread(self, image_data):
        try:
            num_persons = train_face_recognizer(image_data)
            self.master.after(0, lambda: messagebox.showinfo("Thành công", f"Đã train xong {num_persons} người."))
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi khi train model: {str(e)}"))

    def load_students(self):
        self.tree.delete(*self.tree.get_children())
        self.cursor.execute("SELECT msv, name, birthdate, class FROM students")
        for row in self.cursor.fetchall():
            self.tree.insert('', 'end', values=row)

    def add_or_update_student(self):
        msv = self.msv_entry.get()
        name = self.name_entry.get()
        birthdate = self.birthdate_entry.get()
        class_name = self.class_entry.get()

        if not all([msv, name, birthdate, class_name]):
            messagebox.showerror("Lỗi", "Vui lòng điền đầy đủ thông tin")
            return

        try:
            self.cursor.execute('''INSERT OR REPLACE INTO students (msv, name, birthdate, class) 
                                   VALUES (?, ?, ?, ?)''', (msv, name, birthdate, class_name))
            self.conn.commit()
            self.load_students()
            messagebox.showinfo("Thành công", "Đã thêm/cập nhật sinh viên")
            self.clear_entries()
        except sqlite3.Error as e:
            messagebox.showerror("Lỗi", f"Không thể thêm/cập nhật: {e}")

    def delete_student(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showerror("Lỗi", "Vui lòng chọn sinh viên để xóa")
            return

        msv = self.tree.item(selected)['values'][0]
        if messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa sinh viên có MSV {msv}?"):
            try:
                self.cursor.execute("DELETE FROM students WHERE msv=?", (msv,))
                self.cursor.execute("DELETE FROM face_images WHERE msv=?", (msv,))
                self.conn.commit()
                self.load_students()
                messagebox.showinfo("Thành công", "Đã xóa sinh viên")
                self.clear_entries()
            except sqlite3.Error as e:
                messagebox.showerror("Lỗi", f"Không thể xóa: {e}")

    def search_student(self):
        search_term = self.search_entry.get()
        self.cursor.execute("""SELECT msv, name, birthdate, class FROM students 
                               WHERE msv LIKE ? OR name LIKE ?""",
                            ('%' + search_term + '%', '%' + search_term + '%'))
        self.tree.delete(*self.tree.get_children())
        for row in self.cursor.fetchall():
            self.tree.insert('', 'end', values=row)

    def clear_entries(self):
        for entry in [self.msv_entry, self.name_entry, self.birthdate_entry, self.class_entry]:
            entry.delete(0, tk.END)

    def on_tree_select(self, event):
        selected = self.tree.selection()
        if selected:
            values = self.tree.item(selected)['values']
            entries = [self.msv_entry, self.name_entry, self.birthdate_entry, self.class_entry]
            for entry, value in zip(entries, values):
                entry.delete(0, tk.END)
                entry.insert(0, value)

    def view_images(self):
        msv = self.msv_entry.get()
        if not msv:
            messagebox.showerror("Lỗi", "Vui lòng chọn sinh viên để xem ảnh")
            return

        self.cursor.execute("SELECT image FROM face_images WHERE msv=?", (msv,))
        images = self.cursor.fetchall()

        if not images:
            messagebox.showinfo("Thông báo", "Không có ảnh cho sinh viên này")
            return

        image_window = tk.Toplevel(self.master)
        image_window.title(f"Ảnh của sinh viên {msv}")

        canvas = tk.Canvas(image_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(image_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        for i, img_data in enumerate(images):
            img = Image.open(io.BytesIO(img_data[0]))
            img.thumbnail((200, 200))  # Resize image for thumbnail
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(frame, image=photo)
            label.image = photo
            label.grid(row=i // 3, column=i % 3, padx=5, pady=5)

        image_window.update()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def add_image(self):
        msv = self.msv_entry.get()
        if not msv:
            messagebox.showerror("Lỗi", "Vui lòng chọn sinh viên trước khi thêm ảnh")
            return

        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")])
        if not file_paths:
            return  # User canceled file selection

        added_count = 0
        for file_path in file_paths:
            try:
                with Image.open(file_path) as img:
                    img.thumbnail((500, 500))
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format=img.format)
                    img_byte_arr = img_byte_arr.getvalue()

                self.cursor.execute("SELECT COUNT(*) FROM face_images WHERE msv=?", (msv,))
                image_count = self.cursor.fetchone()[0]

                self.cursor.execute("INSERT INTO face_images (msv, image_number, image) VALUES (?, ?, ?)",
                                    (msv, image_count + 1, img_byte_arr))
                self.conn.commit()
                added_count += 1
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể thêm ảnh {file_path}: {str(e)}")

        if added_count > 0:
            messagebox.showinfo("Thành công", f"Đã thêm {added_count} ảnh mới cho sinh viên")
        else:
            messagebox.showwarning("Cảnh báo", "Không có ảnh nào được thêm")


root = tk.Tk()
app = StudentManagementSystem(root)
root.mainloop()