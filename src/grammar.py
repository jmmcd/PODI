#! /usr/bin/env python

# Taken from PonyGE
# Copyright (c) 2009-2012 Erik Hemberg and James McDermott
# Hereby licensed under the GNU GPL v3.
# http://ponyge.googlecode.com

import sys, copy, re, random, math, operator

class Grammar(object):
    """Context Free Grammar"""
    NT = "NT" # Non Terminal
    T = "T" # Terminal

    def __init__(self, file_name):
        self.rules = {}
        self.non_terminals, self.terminals = set(), set()
        self.start_rule = None

        self.read_bnf_file(file_name)

    def read_bnf_file(self, file_name):
        """Read a grammar file in BNF format"""
        rule_separator = "::="
        # Don't allow space in NTs, and use lookbehind to match "<"
        # and ">" only if not preceded by backslash. Group the whole
        # thing with capturing parentheses so that split() will return
        # all NTs and Ts. TODO does this handle quoted NT symbols?
        non_terminal_pattern = r"((?<!\\)<\S+?(?<!\\)>)"
        # Use lookbehind again to match "|" only if not preceded by
        # backslash. Don't group, so split() will return only the
        # productions, not the separators.
        production_separator = r"(?<!\\)\|"

        # Read the grammar file
        for line in open(file_name, 'r'):
            if not line.startswith("#") and line.strip() != "":
                # Split rules. Everything must be on one line
                if line.find(rule_separator):
                    lhs, productions = line.split(rule_separator, 1) # 1 split
                    lhs = lhs.strip()
                    if not re.search(non_terminal_pattern, lhs):
                        raise ValueError("lhs is not a NT:", lhs)
                    self.non_terminals.add(lhs)
                    if self.start_rule == None:
                        self.start_rule = (lhs, self.NT)
                    # Find terminals and non-terminals
                    tmp_productions = []
                    for production in re.split(production_separator, productions):
                        production = production.strip().replace(r"\|", "|")
                        tmp_production = []
                        for symbol in re.split(non_terminal_pattern, production):
                            symbol = symbol.replace(r"\<", "<").replace(r"\>", ">")
                            if len(symbol) == 0:
                                continue
                            elif re.match(non_terminal_pattern, symbol):
                                tmp_production.append((symbol, self.NT))
                            else:
                                self.terminals.add(symbol)
                                tmp_production.append((symbol, self.T))

                        tmp_productions.append(tmp_production)
                    # Create a rule
                    if not lhs in self.rules:
                        self.rules[lhs] = tmp_productions
                    else:
                        raise ValueError("lhs should be unique", lhs)
                else:
                    raise ValueError("Each rule must be on one line")

    def __str__(self):
        return "%s %s %s %s" % (self.terminals, self.non_terminals,
                                self.rules, self.start_rule)
