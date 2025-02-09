import gui_aux as gui


# Create the main window with root object
window_title = 'Data visualization'
window_size = '500x400'
root = gui.create_window(window_title, window_size)

# Create Three entry
width = 10
entry1 = gui.create_entry(root, width)
gui.entry_position_pack(entry1,10,10)
entry2 = gui.create_entry(root, width)
gui.entry_position_pack(entry2,10,10)
entry3 = gui.create_entry(root, width)
gui.entry_position_pack(entry3,10,10)
entry_set = [entry1, entry2, entry3]

# Create Three label and One Result Label
textin = [f'Input{i}' for i in range(1,4)]
font = 'Arial'
font_size = 12
labelset = [gui.create_label(root,textin[i],font,font_size) for i in range(3)]
for i in range(len(labelset)):
    gui.label_position_place(labelset[i],80, 10 + i*40)

# Resule Label
result_label = gui.create_label(root,'Sum of three is ???')
gui.label_position_place(result_label, 100, 200)

# Error Label Set
error_color = 'red'
error_label = gui.create_label(root,'Invalid input!',font,font_size, error_color)

# Create one button to submit the entries input
def calculate_sum():
    sum = 0
    for index, _ in enumerate(entry_set):
        sum += gui.get_entryInput(entry_set[index], error_label)
    gui.change_labelText(result_label,f'Sum of three is {sum}')

button_text = 'Calculate'
button = gui.create_button(root,button_text, calculate_sum)
gui.button_position_place(button, 150,150)


# Run
root.mainloop()

