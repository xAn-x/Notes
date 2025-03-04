### Unix Commands  

| Command | Description                                    |
| ------- | ---------------------------------------------- |
| `ls`    | Lists directory contents                       |
| `cd`    | Changes the current directory                  |
| `pwd`   | Prints the current working directory           |
| `mkdir` | Creates a new directory                        |
| `rmdir` | Removes an empty directory                     |
| `rm`    | Removes files or directories                   |
| `cp`    | Copies files or directories                    |
| `mv`    | Moves or renames files or directories          |
| `touch` | Creates an empty file or updates the timestamp |
| `cat`   | Concatenates and displays file content         |
| `man`   | Displays the manual for a command              |
| `echo`  | Displays a line of text                        |
| `chmod` | Changes file modes or permissions              |
| `chown` | Changes file owner and group                   |
| `ps`    | Displays current active processes              |
| `top`   | Displays system tasks and resource usage       |
| `kill`  | Terminates a process                           |
| `grep`  | Searches for patterns in files                 |
| `find`  | Searches for files in a directory hierarchy    |
| `tar`   | Archives files                                 |
| `gzip`  | Compresses files                               |
| `ssh`   | Securely logs into a remote machine            |
| `scp`   | Securely copies files between hosts            |
| `wget`  | Downloads files from the web                   |
| `curl`  | Transfers data from or to a server             |

---
### Grep and Pattern Matching  

| Command                       | Description                                               |
| ----------------------------- | --------------------------------------------------------- |
| `grep <pattern> FILE`         | Searches for a pattern in FILE and returns matching lines |
| `grep -i <pattern> FILE`      | Performs a case-insensitive search                        |
| `grep -v <pattern> FILE`      | Inverts the match, returning non-matching lines           |
| `grep -r <pattern> DIR`       | Recursively searches for a pattern in a directory         |
| `grep -E <pattern> FILE`      | Uses extended regex for pattern matching                  |
| `grep -o <pattern> FILE`      | Prints only the matched parts of the line                 |
| `grep -n <pattern> FILE`      | Displays line numbers of matching lines                   |
| `grep -c <pattern> FILE`      | Displays the count of matching lines                      |
| `grep -q <pattern> FILE`      | Runs quietly, returning status only (0 if found)          |
| `grep --color <pattern> FILE` | Highlights matching patterns in output                    |

---
### Useful Patterns in Grep  

| Pattern  | Description                                       |
| -------- | ------------------------------------------------- |
| `.*`     | Matches any character zero or more times          |
| `^`      | Matches the start of a line                       |
| `$`      | Matches the end of a line                         |
| `[abc]`  | Matches any single character listed (a, b, or c)  |
| `[^abc]` | Matches any character not listed (not a, b, or c) |
| `a?`     | Matches a zero or one occurrence of a             |
| `a*`     | Matches a zero or more occurrences of a           |
| `a+`     | Matches one or more occurrences of a              |
| `a{n}`   | Matches exactly n occurrences of a                |
| `a{n,}`  | Matches at least n occurrences of a               |
| `a{n,m}` | Matches between n and m occurrences of a          |

 ---

# Vim Commands Cheat Sheet  

### Basic Commands  

| Command                | Description                                          |  
|-----------------------|------------------------------------------------------|  
| `vim filename`        | Open a file named `filename`                         |  
| `i`                   | Enter insert mode (before the cursor)                |  
| `I`                   | Enter insert mode (at the beginning of the line)     |  
| `a`                   | Enter insert mode (after the cursor)                 |  
| `A`                   | Enter insert mode (at the end of the line)           |  
| `o`                   | Open a new line below the current line and enter insert mode |  
| `O`                   | Open a new line above the current line and enter insert mode |  
| `Esc`                 | Exit insert mode                                     |  
| `:w`                  | Save the current file                               |  
| `:q`                  | Quit Vim                                           |  
| `:wq`                 | Save and quit Vim                                   |  
| `:q!`                 | Quit without saving                                 |  

### Navigation Commands  

| Command    | Description                     |     |
| ---------- | ------------------------------- | --- |
| `h`        | Move left                       |     |
| `j`        | Move down                       |     |
| `k`        | Move up                         |     |
| `l`        | Move right                      |     |
| `gg`       | Go to the beginning of the file |     |
| `G`        | Go to the end of the file       |     |
| `:n`       | Go to line number `n`           |     |
| `Ctrl + f` | Scroll forward one screen       |     |
| `Ctrl + b` | Scroll backward one screen      |     |
| `0`        | Go to the beginning of the line |     |
| `$`        | Go to the end of the line       |     |

### Editing Commands  

| Command    | Description                                          |     |
| ---------- | ---------------------------------------------------- | --- |
| `x`        | Delete the character under the cursor                |     |
| `dw`       | Delete from the cursor to the start of the next word |     |
| `d$`       | Delete from the cursor to the end of the line        |     |
| `d^`       | Delete from the cursor to the beginning of the line  |     |
| `dd`       | Delete the current line                              |     |
| `yy`       | Copy (yank) the current line                         |     |
| `p`        | Paste the copied or deleted text after the cursor    |     |
| `u`        | Undo the last action                                 |     |
| `Ctrl + r` | Redo the last undone action                          |     |
| `~`        | Change the case of the character under the cursor    |     |

### Searching and Replacing  

| Command             | Description                                                     |     |
| ------------------- | --------------------------------------------------------------- | --- |
| `/pattern`          | Search forward for `pattern`                                    |     |
| `?pattern`          | Search backward for `pattern`                                   |     |
| `n`                 | Repeat the last search in the same direction                    |     |
| `N`                 | Repeat the last search in the opposite direction                |     |
| `:%s/old/new/g`     | Replace all occurrences of `old` with `new` in the whole file   |     |
| `:s/old/new/g`      | Replace all occurrences of `old` with `new` in the current line |     |
| `:set ignorecase`   | Ignore case in searches                                         |     |
| `:set noignorecase` | Disregard ignoring case in searches                             |     |
|                     |                                                                 |     |

### File Management  

| Command         | Description                                      |     |
| --------------- | ------------------------------------------------ | --- |
| `:e filename`   | Open a file named `filename`                     |     |
| `:n`            | Switch to the next file in the argument list     |     |
| `:N`            | Switch to the previous file in the argument list |     |
| `:set number`   | Display line numbers                             |     |
| `:set nonumber` | Hide line numbers                                |     |

### Miscellaneous Commands  

| Command           | Description                  |     |
| ----------------- | ---------------------------- | --- |
| `:help`           | Open the help menu           |     |
| `:set autoindent` | Enable automatic indentation |     |
| `:set syntax=on`  | Enable syntax highlighting   |     |
| `:syntax off`     | Disable syntax highlighting  |     |
| `Ctrl + w, w`     | Switch between open windows  |     |
| `:tabnew`         | Open a new tab               |     |
| `:tabnext`        | Move to the next tab         |     |
| `:tabprev`        | Move to the previous tab     |     |
