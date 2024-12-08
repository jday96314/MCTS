import json
import re

def MinifyRules(lud_rules: str) -> str:
    lud_rules = lud_rules.replace('\\n', '')
    lud_rules = lud_rules.replace('\\t', '')

    lud_rules = lud_rules.replace('\r', '')
    lud_rules = lud_rules.replace('\n', ' ')
    lud_rules = lud_rules.replace('\t', '')
    for _ in range(20):
        lud_rules = lud_rules.replace('  ', ' ')

    return lud_rules

# After first opening parenthesis, everything in each subsequent pair of 2nd level parentheses is split onto
# a new line and indented by 1 tab.
# If one of the resulting lines is over 200 characters long (ignoring whitespace), it is recursively
# unminified and broken onto additional lines.
# Originally written to unminify sections within the Ludii game rules, but seems to work well for
# unminifying full Ludii game descriptions too.
def UnminifyRules(section: str) -> str:
    unminified_section = ''

    # Exit early for sections that can't be further deminified.
    # Anything within a single pair of parentheses is already maximally-unminified.
    if section.count('(') == 1:
        return section

    # Record first line.
    current_line = section[0]
    current_input_char_index = 1
    while section[current_input_char_index] != '(':
        current_line += section[current_input_char_index]
        current_input_char_index += 1

    unminified_section += current_line.rstrip(' ') + '\n'

    # Record subsequent lines.
    current_line = '\t'
    opening_paren_count = 0
    closing_paren_count = 0
    while ((opening_paren_count == 0) or (opening_paren_count != closing_paren_count)) and (current_input_char_index < len(section)):
        if not ((current_line == '\t') and (section[current_input_char_index] == ' ')):
            current_line += section[current_input_char_index]
        
        if section[current_input_char_index] == '(':
            opening_paren_count += 1
        elif section[current_input_char_index] == ')':
            closing_paren_count += 1

        if (opening_paren_count == closing_paren_count) and (section[current_input_char_index] == ')'):
            if len(current_line.strip()) > 200:
                unminified_line = UnminifyRules(current_line.strip())
                indented_unminified_line = '\t' + '\n\t'.join(unminified_line.split('\n'))
                unminified_section += indented_unminified_line + '\n'
            else:
                unminified_section += current_line.rstrip(' ') + '\n'
            
            current_line = '\t'
            opening_paren_count = 0
            closing_paren_count = 0

        current_input_char_index += 1

    # Record final line.
    unminified_section += current_line.rstrip(' ').lstrip('\t')

    return unminified_section

if __name__ == '__main__':
    with open('data/rules_to_pca_utilities.json', 'r') as test_data_file:
        minified_test_rules = list(json.load(test_data_file).keys())

    for minified_test_rule in minified_test_rules:
        unminified_test_rule = UnminifyRules(minified_test_rule)

        # Space inconsistency is OK, we're more concerned with tabs and newlines.
        assert minified_test_rule.replace(' ', '') == MinifyRules(unminified_test_rule).replace(' ', '')
        # assert minified_test_rule == MinifyRules(unminified_test_rule)

    print('All tests passed.')