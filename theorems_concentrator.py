import re


class LyxParser:
    def __init__(self, file_name):
        self.i = 0
        self.file_name = file_name
        self.file = open(file_name, encoding="utf8")
        self.lines = self.file.readlines()
        self.n = len(self.lines)
        self.post_deeper = 0
        self.begin_layot_pattern = re.compile(r'\\begin_layout\s(Theorem|Claim|Definition|Lemma$)')
        self.last_line_pattern = re.compile(r'\n.*\n$')
        self.end_body_pattern = re.compile(r'end_body')
        self.no_extension_pattern = re.compile('(.*)\.lyx')
        self.reference_name_pattern = re.compile('name (\".*\")\n')
        # self.last_end_deeper_pattern = re.compile(r'(.*)\n\\end_deeper$',  re.MULTILINE|re.DOTALL)
        self.theorems_dict = {'Claim': '\\color green\n טענה',
                              'Theorem': '\\color brown\n משפט',
                              'Definition': '\\color teal\n הגדרה',
                              'Lemma': '\\color purple\n למה'
                              }


    def create_parsed_file(self):
        # Create a second file in the same directory with the theorems list at the buttom
        file_with_theorems = self.unify_original_with_theorems()
        file_name_no_extension = self.no_extension_pattern.match(self.file_name)
        new_file_name = file_name_no_extension.group(1) + ' עם רשימת משטים.lyx'
        with open(new_file_name, 'w', encoding="utf8") as new_file:
            new_file.write(file_with_theorems)


    def unify_original_with_theorems(self):
        # Creates a text variable of the edited file:
        enumerated_theorems = self.generate_enumerated_theorems_latex()
        self.file.seek(0)
        file_text = self.file.read()
        file_with_theorems = self.safe_replace(self.end_body_pattern, enumerated_theorems, file_text)
        return file_with_theorems


    def generate_enumerated_theorems_latex(self):
        # convert a list of theorems to enumerated enviorment of latex:
        theorems = self.extract_theorems()
        enumerated_theorems = ''
        for theorem_tuple in theorems:
            enumerated_theorems = enumerated_theorems + self.theorem_to_enumerated(*theorem_tuple)
        enumerated_theorems = enumerated_theorems + '\\end_body'
        return enumerated_theorems


    def theorem_to_enumerated(self, theorem_type, theorem_text):
        # Convert a theorem to enumerated format:
        opening = '\\begin_layout Enumerate\n\\series bold\n' + \
                   self.theorems_dict[theorem_type] + \
                  ': \n\\series default\n\\color inherit\n'
        return opening + theorem_text


    def extract_theorems(self):
        # Scans the document for theorems and stores them in a list:
        theorems = []
        while self.i < self.n:
            current_line = self.readline()
            theorem_type = re.search(self.begin_layot_pattern, current_line)
            if theorem_type is not None:
                theorems.append((theorem_type.group(1), self.parse_theorem(theorem_type.group(1))))
        return theorems


    def parse_theorem(self, layout_type):
        new_row_format = '\\begin_layout {0}\n'.format(layout_type)
        theorem_text = ''
        while True:
            current_line = self.readline()
            if current_line == "\\begin_inset CommandInset label\n":
                self.i += 1
                theorem_text += self.label_to_reference()
                continue
            elif current_line != '\\end_layout\n' and not (self.post_deeper and current_line == new_row_format):
                theorem_text = theorem_text + current_line
            else:
                next_line = self.readline()
                next_next_line = self.readline()

                if next_line == '\n' and next_next_line == '\\begin_deeper\n':
                    theorem_text += '\\end_layout\n\n\\begin_deeper\n'
                    nested_text_value = self.parse_nested_text()
                    self.post_deeper = 1
                    theorem_text += nested_text_value
                    # theorem_text = self.safe_replace(self.last_line_pattern, nested_text_value, theorem_text)
                    if self.readline() == '\\begin_layout {0}\n'.format(layout_type):
                        self.i -= 1
                    else:
                        self.i -= 1
                        return theorem_text + '\\end_deeper\n'


                elif next_next_line == new_row_format or self.post_deeper and current_line == new_row_format:
                    # self.i -= 1
                    if self.post_deeper and current_line == new_row_format:
                        self.i -= 2
                        current_hirarchy_post_deeper = 1
                    further_text = self.parse_theorem(layout_type)
                    try:
                        if current_hirarchy_post_deeper:
                            further_text = '\n\\begin_layout Standard\n' +\
                                           further_text +\
                                           '\\end_layout\n\n\\end_deeper\n'
                            self.post_deeper = 0
                    except:
                        further_text = '\\begin_inset Newline newline\n\\end_inset\n\n' + further_text
                    return theorem_text + further_text

                else:
                    self.i -= 2
                    try:
                        if current_hirarchy_post_deeper:
                            theorem_text += '\\end_deeper\n'
                    except:
                        self.post_deeper = 0
                        theorem_text += '\\end_layout\n\n'
                    return theorem_text


    def parse_nested_text(self):
        nested_text_value = ''
        depth = 1
        while True:
            current_line = self.readline()

            if current_line == "\\begin_inset CommandInset label":
                self.i += 1
                nested_text_value += self.label_to_reference()
                continue


            nested_text_value += current_line
            if current_line == '\\begin_deeper\n':
                depth += 1
            elif current_line == '\\end_deeper\n':
                depth -= 1

            if depth == 0:
                return self.safe_replace(self.last_line_pattern, '\n', nested_text_value)


    def readline(self):
        self.i += 1
        return self.lines[self.i - 1]


    def label_to_reference(self):
        current_line = self.readline()
        reference_name = re.match(self.reference_name_pattern, current_line)
        self.i += 2
        return self.generate_reference(reference_name.group(1))


    @staticmethod
    def generate_reference(reference_name):
        return '\\begin_inset CommandInset ref\nLatexCommand eqref\nreference {0}\nplural "false"' \
               '\ncaps "false"\nnoprefix "false"\n\n\\end_inset\n'.format(reference_name)


    @staticmethod
    def safe_replace(pattern, replacer, text):
        replacer = re.sub(r'\\', r'GiladHaHomoBatachat', replacer)
        text = re.sub(r'\\', r'GiladHaHomoBatachat', text)
        replaced = re.sub(pattern, replacer, text)
        replaced_unescaped = re.sub(r'GiladHaHomoBatachat', r'\\', replaced)
        return replaced_unescaped


# %%
# file_name = r'C:\Users\mkali\Google Drive\University\Current Semester\Computational Physics\Notes Example.lyx'
# file_name = r'C:\Users\mkali\Google Drive\University\Current Semester\Set Theory\קבוצות - סיכומי שיעור.lyx'
# file_name = r'C:\Users\mkali\Google Drive\University\Current Semester\Infinitesimal Calculus\אינפי - סיכומי שיעור - למשחקים.lyx'
file_name = r'C:\Users\mkali\Google Drive\University\Previous Semesters\2 - First Year - Second Semester\Math\Infinitesimal II\אינפי - ריכוז משפטים.lyx'
L = LyxParser(file_name)
L.create_parsed_file()
L.file.close()

