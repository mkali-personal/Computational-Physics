import re
# Read in the file

file_name = r'C:\Users\mkali\Google Drive\University\Current Semester\Computational Physics\Week 7\comp_latex.lyx'#input('Insert lyx file path to parse:')

with open(file_name, 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace('\n\\begin_inset Formula', '')
filedata = filedata.replace('\n\\end_inset\n\n', '')
filedata = filedata.replace('\\[\n', '$$')
filedata = filedata.replace('\n\\]', '$$')
filedata = filedata.replace('\\end_layout', '')
filedata = filedata.replace('\\begin_layout Subsection\n', '## ')
filedata = filedata.replace('\\begin_layout Section\n', '# ')
filedata = filedata.replace('\\begin_layout Standard \n', '')
filedata = filedata.replace('\\begin_layout Standard\n', '')

# Cut away header and footer of the content:
regex_begin = re.compile('.*today\n*', re.MULTILINE|re.DOTALL)
regex_end = re.compile(r'\n*\\end_body.*', re.MULTILINE|re.DOTALL)

filedata = re.sub(regex_begin, '', filedata)
filedata = re.sub(regex_end, '', filedata)

no_extension_re = re.compile('(.*)\.lyx')
file_name_no_extension = no_extension_re.match(file_name)
txt_file_name = file_name_no_extension.group(1) + '.txt'


# Write the file out again
with open(txt_file_name, 'w') as file:
  file.write(filedata)
