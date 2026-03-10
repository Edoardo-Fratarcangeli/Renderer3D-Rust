import sys
import os
import subprocess
import threading
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QPushButton, QLabel,
    QSplitter, QFrame, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QBrush

# Signal helper for thread-safe UI updates
class LogSignal(QObject):
    log_message = pyqtSignal(str, str)

class TestManagerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧪 Rust Test Manager")
        self.setGeometry(100, 100, 1000, 700)
        
        # Selection state
        self.selected_items = {}
        
        # Signal for thread-safe logging
        self.log_signal = LogSignal()
        self.log_signal.log_message.connect(self.append_log)
        
        # Apply dark theme stylesheet
        self.setStyleSheet(self.get_stylesheet())
        
        # Setup UI
        self.setup_ui()
        
        # Load tests
        self.tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.populate_tree()

    def get_stylesheet(self):
        return """
            QMainWindow {
                background-color: #0d1117;
            }
            QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: 'Segoe UI', Arial;
            }
            QSplitter::handle {
                background-color: #21262d;
                width: 2px;
            }
            QTreeWidget {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QTreeWidget::item {
                padding: 6px 4px;
                border-radius: 4px;
                background-color: transparent;
            }
            QTreeWidget::item:hover {
                background-color: #21262d;
            }
            QTreeWidget::item:selected {
                background-color: transparent;
                outline: none;
                color: #c9d1d9;
            }
            QTreeWidget::item:focus {
                background-color: transparent;
                outline: none;
                border: none;
            }
            QTreeWidget::branch {
                background-color: transparent;
            }
            QTreeWidget::branch:selected {
                background-color: transparent;
            }
            QTreeWidget::branch:hover {
                background-color: transparent;
            }
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {
                image: url(none);
                border-image: none;
            }
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {
                image: url(none);
                border-image: none;
            }
            QTreeWidget:focus {
                outline: none;
                border: 1px solid #30363d;
            }
            QTreeWidget::item:selected:active {
                background-color: transparent;
            }
            QTreeWidget::item:selected:!active {
                background-color: transparent;
            }
            QTreeWidget::indicator {
                width: 18px;
                height: 18px;
            }
            QTreeWidget::indicator:unchecked {
                border: 2px solid #484f58;
                border-radius: 4px;
                background-color: transparent;
            }
            QTreeWidget::indicator:checked {
                border: 2px solid #58a6ff;
                border-radius: 4px;
                background-color: #58a6ff;
            }
            QTextEdit {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 12px;
                color: #c9d1d9;
            }
            QPushButton {
                background-color: #238636;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
            QPushButton:pressed {
                background-color: #1a7f37;
            }
            QPushButton#secondaryBtn {
                background-color: #21262d;
                border: 1px solid #30363d;
            }
            QPushButton#secondaryBtn:hover {
                background-color: #30363d;
            }
            QLabel#headerLabel {
                font-size: 16px;
                font-weight: bold;
                color: #f0f6fc;
                padding: 8px 0;
            }
            QLabel#subLabel {
                font-size: 12px;
                color: #8b949e;
            }
            QFrame#card {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
        """

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left Panel
        left_panel = QFrame()
        left_panel.setObjectName("card")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)
        
        # Header
        header = QLabel("📁 Test Suite")
        header.setObjectName("headerLabel")
        left_layout.addWidget(header)
        
        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        
        select_all_btn = QPushButton("✓ Select All")
        select_all_btn.setObjectName("secondaryBtn")
        select_all_btn.clicked.connect(self.select_all)
        btn_row.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("✗ Deselect All")
        deselect_all_btn.setObjectName("secondaryBtn")
        deselect_all_btn.clicked.connect(self.deselect_all)
        btn_row.addWidget(deselect_all_btn)
        
        btn_row.addStretch()
        left_layout.addLayout(btn_row)
        
        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(20)
        self.tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Remove focus rectangle
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)  # Disable selection
        self.tree.itemChanged.connect(self.on_item_changed)
        left_layout.addWidget(self.tree)
        
        # Run button
        run_btn = QPushButton("▶  Run Selected Tests")
        run_btn.clicked.connect(self.run_tests)
        left_layout.addWidget(run_btn)
        
        splitter.addWidget(left_panel)
        
        # Right Panel
        right_panel = QFrame()
        right_panel.setObjectName("card")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)
        
        # Output header
        output_header = QLabel("📋 Output")
        output_header.setObjectName("headerLabel")
        right_layout.addWidget(output_header)
        
        # Output text
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        right_layout.addWidget(self.output)
        
        # Clear button
        clear_btn = QPushButton("🗑 Clear Output")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.clicked.connect(lambda: self.output.clear())
        right_layout.addWidget(clear_btn)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 650])

    def populate_tree(self):
        self.tree.clear()
        self.selected_items.clear()
        
        # Block signals during population
        self.tree.blockSignals(True)
        
        # Root
        root_item = QTreeWidgetItem(self.tree, ["📦 tests"])
        root_item.setFlags(root_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
        root_item.setCheckState(0, Qt.CheckState.Unchecked)
        root_item.setData(0, Qt.ItemDataRole.UserRole, ("root",))
        root_item.setExpanded(True)
        
        # Walk directories
        for item in sorted(os.listdir(self.tests_dir)):
            path = os.path.join(self.tests_dir, item)
            if os.path.isdir(path) and item != "__pycache__":
                # Category
                cat_item = QTreeWidgetItem(root_item, [f"📁 {item}"])
                cat_item.setFlags(cat_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
                cat_item.setCheckState(0, Qt.CheckState.Unchecked)
                cat_item.setData(0, Qt.ItemDataRole.UserRole, ("category", item))
                cat_item.setExpanded(True)
                
                # Test files
                for file in sorted(os.listdir(path)):
                    if file.endswith(".rs") and file != "main.rs":
                        mod_name = file.replace(".rs", "")
                        test_item = QTreeWidgetItem(cat_item, [f"📄 {mod_name}"])
                        test_item.setFlags(test_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        test_item.setCheckState(0, Qt.CheckState.Unchecked)
                        test_item.setData(0, Qt.ItemDataRole.UserRole, ("test", item, mod_name))
        
        self.tree.blockSignals(False)

    def on_item_changed(self, item, column):
        # Qt handles tristate automatically with ItemIsAutoTristate flag
        pass

    def select_all(self):
        self.tree.blockSignals(True)
        self.set_check_state_recursive(self.tree.invisibleRootItem(), Qt.CheckState.Checked)
        self.tree.blockSignals(False)

    def deselect_all(self):
        self.tree.blockSignals(True)
        self.set_check_state_recursive(self.tree.invisibleRootItem(), Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)

    def set_check_state_recursive(self, item, state):
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            self.set_check_state_recursive(child, state)

    def get_selected_tests(self):
        """Returns list of commands to run based on selection"""
        commands = []
        categories_to_run = set()
        tests_to_run = {}  # category -> set of modules
        
        def walk(item):
            for i in range(item.childCount()):
                child = item.child(i)
                if child.checkState(0) == Qt.CheckState.Checked:
                    data = child.data(0, Qt.ItemDataRole.UserRole)
                    if data:
                        if data[0] == "root":
                            walk(child)
                        elif data[0] == "category":
                            categories_to_run.add(data[1])
                        elif data[0] == "test":
                            cat, mod = data[1], data[2]
                            if cat not in categories_to_run:  # Don't add if whole category checked
                                if cat not in tests_to_run:
                                    tests_to_run[cat] = set()
                                tests_to_run[cat].add(mod)
                elif child.checkState(0) == Qt.CheckState.PartiallyChecked:
                    walk(child)  # Recurse to find checked children
                    
        walk(self.tree.invisibleRootItem())
        
        # Build commands
        for cat in categories_to_run:
            commands.append(["cargo", "test", "--test", cat])
            
        for cat, mods in tests_to_run.items():
            if cat not in categories_to_run:
                for mod in mods:
                    commands.append(["cargo", "test", "--test", cat, mod])
        
        return commands

    def run_tests(self):
        commands = self.get_selected_tests()
        
        if not commands:
            self.append_log("⚠️ No tests selected!\n", "error")
            return
        
        self.output.clear()
        self.append_log(f"🚀 Running {len(commands)} test command(s)...\n\n", "info")
        
        # Run in thread
        threading.Thread(target=self.execute_commands, args=(commands,), daemon=True).start()

    def execute_commands(self, commands):
        try:
            total_suites = len(commands)
            total_passed = 0
            total_failed = 0
            any_process_failed = False
            
            for cmd in commands:
                full_cmd = " ".join(cmd)
                self.log_signal.log_message.emit(f"{'─'*60}\n$ {full_cmd}\n\n", "dim")
                
                process = subprocess.Popen(
                    ["pwsh", "-c", full_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(self.tests_dir),
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                for line in process.stdout:
                    self.log_signal.log_message.emit(line, "info")
                    
                    # Parse cargo test result summary
                    if "test result:" in line:
                        match = re.search(r'(\d+)\s+passed;\s+(\d+)\s+failed', line)
                        if match:
                            total_passed += int(match.group(1))
                            total_failed += int(match.group(2))
                
                process.wait()
                if process.returncode != 0:
                    any_process_failed = True

            # Final summary report
            self.log_signal.log_message.emit(f"\n{'━'*60}\n", "dim")
            self.log_signal.log_message.emit("📊  TEST RUN SUMMARY\n", "info")
            self.log_signal.log_message.emit(f"{'━'*60}\n", "dim")
            
            self.log_signal.log_message.emit(f"Total Suites: {total_suites}\n", "info")
            
            total_tests = total_passed + total_failed
            self.log_signal.log_message.emit(f"Total Tests:  {total_tests}\n", "info")
            self.log_signal.log_message.emit(f"Passed:       {total_passed}\n", "success")
            
            if total_failed > 0:
                self.log_signal.log_message.emit(f"Failed:       {total_failed}\n", "error")
            else:
                self.log_signal.log_message.emit(f"Failed:       0\n", "info")
                
            self.log_signal.log_message.emit(f"{'━'*60}\n", "dim")
            
            if any_process_failed or total_failed > 0:
                self.log_signal.log_message.emit("\n❌ Finished with errors\n\n", "error")
            else:
                self.log_signal.log_message.emit("\n✅ Finished successfully\n\n", "success")
                
        except Exception as e:
            self.log_signal.log_message.emit(f"❌ Error: {str(e)}\n\n", "error")

    def append_log(self, text, tag="info"):
        cursor = self.output.textCursor()
        
        fmt = QTextCharFormat()
        if tag == "success":
            fmt.setForeground(QBrush(QColor("#7ee787")))
        elif tag == "error":
            fmt.setForeground(QBrush(QColor("#f85149")))
        elif tag == "dim":
            fmt.setForeground(QBrush(QColor("#8b949e")))
        else:
            fmt.setForeground(QBrush(QColor("#c9d1d9")))
        
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text, fmt)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TestManagerApp()
    window.show()
    sys.exit(app.exec())
