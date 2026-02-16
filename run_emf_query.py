import os
import psycopg2
import psycopg2.extras
import tabulate
from dotenv import load_dotenv
from collections import deque, defaultdict

def aggregation_handler(old, new, operation, isNew=False):
    # Handles the aggregation of grouping variables
    # if isNew implies its the first record to be used for mf table and is handled differently for initialization
    if operation=='SUM' or operation=='COUNT':
        if operation=='COUNT':
            new = 1
        if isNew:
            return new
        return old+new
        
    if operation=='MIN':
        if isNew:
            return new
        if new<old:
            return new
        return old
        
    if operation=='MAX':
        if isNew:
            return new
        if new>old:
            return new
        return old
    if operation=='AVG':
        if isNew:
            return [new,1]
        return [ old[0]+new , old[1]+1]

def split_aggregation_string(input_str):
    # Split by underscore
    parts = input_str.split('_')
    
    # Extract the aggregation (first part in uppercase) this will be used in aggregation_handler
    aggregation = parts[0].upper()
    
    # Extract the number (second part) this is the g.v
    gv = parts[1]
    
    # Extract the remaining text (third part in lowercase) this is the column name
    remaining_text = parts[2].lower()
    
    return aggregation, gv, remaining_text

def parse_expression(expr,mf_val=[],vf=[]):
    # we send mf_table row here and use values in there to replace every variable to its value 
    # after all variable/column names are replaced with their actual values we perform calculation
    # by smartly using try eval() we make this function versatile and it returns calculated final values or the single field
    for item in vf[::-1]:
        if item in expr:
            if mf_val[vf.index(item)] is not None:
                b = mf_val[vf.index(item)]
                # lazy computation for average aggregation
                if type(b)==type([]):
                    b = b[0]/b[1]
            else:
                b = 0
            expr = expr.replace(item,str(b))
    expr = expr.strip().strip("'")   
    try:
        result = eval(expr)
    except ZeroDivisionError:
        result = float('inf')
    except Exception:
        # If non number expression return string
        result = expr
    return result

def to_number(value):
    # we try converting to float, string or leave it unchanged for matching data types when comparing them
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            float(value)
            return float(value)
        except ValueError:
            return value
    return value

def conditionChecker(condition, row):
    # conditionChecker is for MF conditions
    # List of possible operators
    operators = ['>=', '<=', '<>', '=', '>', '<']
    
    # Find which operator is in the condition
    for op in operators:
        if op in condition:
            parts = condition.split(op)
            lhs = parts[0].strip().strip("'") 
            rhs = parts[1].strip().strip("'")
            break
    
    # Get the value from the row
    value = row[lhs]

    #to_number attempts to convert to float if possible, otherwise returns data as string, ensuring both lhs and rhs are of same type
    
    lhs = to_number(lhs)
    rhs = to_number(rhs)
    
    if value is None:
        return False
        
    if op == '=':
        return value == rhs
    elif op == '<>':
        return value != rhs
    elif op == '>=':
        return value >= rhs
    elif op == '<=':
        return value <= rhs
    elif op == '>':
        return value > rhs
    elif op == '<':
        return value < rhs

    return False  # if some data mismatch or something like invalid operator happens

def conditionCheckerHaving(condition, row, combined_VF):
    # this is conditionChecker for having
    
    # List of possible operators
    operators = ['>=', '<=', '<>', '=', '>', '<']
    # Find which operator is in the condition
    for op in operators:
        if op in condition:
            parts = condition.split(op)
            lhs = parts[0].strip().strip("'") 
            rhs = parts[1].strip().strip("'")
            break

    rhs = parse_expression(rhs,row,combined_VF)
    
    for item in combined_VF[::-1]:
        if item in lhs:
            lhs = row[combined_VF.index(item)]
            break
    # lazy computation for average aggregation
    if type(lhs)==type([]) and  len(lhs)==2:
        lhs = lhs[0]/lhs[1]  
    # to number data type match if possible 
    lhs = to_number(lhs)
    rhs = to_number(rhs)

    # condition check with special handling for some relations between none
    if op == '=':
        if rhs is None or lhs is None:
            return False
        return lhs == rhs
    elif op == '<>':
        if lhs is None:
            return False
        elif rhs is None:
            return True
        return lhs != rhs
    elif op == '>=':
        if lhs is None:
            return False
        elif rhs is None:
            return True
        return lhs >= rhs
    elif op == '<=':
        if rhs is None or lhs is None:
            return False
        return lhs <= rhs
    elif op == '>':
        if lhs is None:
            return False
        elif rhs is None:
            return True
        return lhs > rhs
    elif op == '<':
        if rhs is None or lhs is None:
            return False
        return lhs < rhs

    return False  # if some data mismatch or something like invalid operator happens

def conditionCheckerUltimate(expression,row,vf,mf_val):
    # conditionCheckerUltimate is condition checker for EMF, called ultimate as it was much tougher than other 2 condition checkers
    
    # expression could be of 3 types
    # case 1 'gv.attribute' op 'attribute'          <- this is the only one with the pattern 'gv.'
    # case 2 'attribute' op 'aggregate on other gv' <- this is the only one with '_'  
    # case 3 'attribute' op 'literal'               <- this is the only one without either 'gv.' and '_'
    
    case = -1
    for item in vf:
        # print(item)
        if f".{item}" in expression:
            case = 1
            break
    for item in vf:
        if f"_" in item and item in expression:
            case = 2
            break
    if case == -1:
        case = 3

    # List of possible operators
    operators = ['>=', '<=', '<>', '=', '>', '<']
    
    # Find which operator is in the condition
    for op in operators:
        if op in expression:
            parts = expression.split(op)
            lhs = parts[0].strip().strip("'") 
            rhs = parts[1].strip().strip("'")
            break
 
    if case == 1:
        for item in vf:
            if item in lhs:
                lhs = row[item]
                break
        rhs = parse_expression(rhs,mf_val,vf)
    if case == 2:
        lhs = row[lhs]
        rhs = parse_expression(rhs,mf_val,vf)
    if case == 3:
        # case 3 degenerates to conditionChecker as this is an MF type condition
        return conditionChecker(expression,row)
    
    # lazy computation of average aggregation
    if type(rhs)==type([]) and  len(rhs)==2:
        rhs = rhs[0]/rhs[1]  
    if type(lhs)==type([]) and  len(lhs)==2:
        lhs = lhs[0]/lhs[1]  

    # to_number to ensure data type matching
    lhs = to_number(lhs)
    rhs = to_number(rhs)

    # condition checking with special handling for none
    if op == '=':
        if rhs is None or lhs is None:
            return False
        return lhs == rhs
    elif op == '<>':
        if lhs is None:
            return False
        elif rhs is None:
            return True
        return lhs != rhs
    elif op == '>=':
        if lhs is None:
            return False
        elif rhs is None:
            return True
        return lhs >= rhs
    elif op == '<=':
        if rhs is None or lhs is None:
            return False
        return lhs <= rhs
    elif op == '>':
        if lhs is None:
            return False
        elif rhs is None:
            return True
        return lhs > rhs
    elif op == '<':
        if rhs is None or lhs is None:
            return False
        return lhs < rhs

def recursive_condition(condition_str,l,r,row,toggle=1,combined_VF=[],mf_val=[]):
    # this recursive function may look complicated but it simply is a logical solver for conjuntions and disjunctions
    # the i and j variables capture each block beginning and end and sub recursive loops are called if i and j are a nested block
    # if i and j capture a single condition block we call one of the condition checker functions, this is the base case
    # which condition checker to call is provided by the toggle, for example if we need condition checker for having, toggle is 1
    i = l
    j = 0
    ans = True
    and_or = 0
    while i<r:
        # print(i)
        if and_or == 1 and ans==False:
            return False
        if and_or == 2 and ans==True:
            return True
        if condition_str[i]==' ':
            i=i+1
            continue
        if condition_str[i]=='(':
            open=1
            j=i
            while open!=0:
                j=j+1
                if condition_str[j]=='(':
                    open = open+1
                if condition_str[j]==')':
                    open = open-1  
            i=i+1
            j=j-1
            if and_or == 0:
                ans = recursive_condition(condition_str,i,j,row,toggle,combined_VF,mf_val)
            if and_or == 1:
                ans = ans and recursive_condition(condition_str,i,j,row,toggle,combined_VF,mf_val)
            if and_or == 2:
                ans = ans or recursive_condition(condition_str,i,j,row,toggle,combined_VF,mf_val)
            i=j+2
            continue
        if condition_str[i]=="'":
            j = i
            expression = ""
            while j!=r+1 and condition_str[j]!=" ":
                expression = expression+condition_str[j]
                j=j+1
            if toggle==1:
                if and_or == 0:
                    ans = conditionCheckerHaving(expression,row,combined_VF)
                if and_or == 1:
                    ans = ans and conditionCheckerHaving(expression,row,combined_VF)
                if and_or == 2:
                    ans = ans or conditionCheckerHaving(expression,row,combined_VF)
            if toggle==0:
                if and_or == 0:
                    ans = conditionChecker(expression,row)
                if and_or == 1:
                    ans = ans and conditionChecker(expression,row)
                if and_or == 2:
                    ans = ans or conditionChecker(expression,row)
            if toggle==2:
                if and_or == 0:
                    ans = conditionCheckerUltimate(expression,row,combined_VF,mf_val)
                if and_or == 1:
                    ans = ans and conditionCheckerUltimate(expression,row,combined_VF,mf_val)
                if and_or == 2:
                    ans = ans or conditionCheckerUltimate(expression,row,combined_VF,mf_val)
            i=j+1
            continue
        # we handle 'and' and 'or' operators here, 'xor' is not handled
        if condition_str[i]=='a':
            and_or = 1
            i = i+3
            continue
        if condition_str[i]=='o':
            and_or = 2
            i = i+2
            continue
    return ans

def build_dependency_graph(P,V):
    # we build dependency graph, here graph is simply a directed graph, if a needs b, then we add edge b->a
    # we also keep track of each vertex(gv) indegree, which tells count of other gv it needs calculated first
    graph = {}
    indegree = {}
    for gv in P:
        graph[gv] = []
        indegree[gv] = 0
        
    for gv, conditions in P.items():
        if gv == '0':
            continue
        for condition in conditions:
            for other_gv in P:
                if other_gv == '0':
                    for v in V:
                        if f".{v}'" in condition:
                            graph[other_gv].append(gv)
                            indegree[gv] += 1
                else:
                    if f"_{other_gv}_" in condition and other_gv != gv:
                        graph[other_gv].append(gv)
                        indegree[gv] += 1
                
    return graph, indegree
    return graph

def level_order_toposort(graph, indegree):
    # this is simply topological sort but we maintain layers of dependency, 
    # if a and b need c, then a,b are in same level but after level of c 
    
    level_map = defaultdict(list)
    queue = deque()
    
    # Start with 0-indegree nodes
    for gv in indegree:
        if indegree[gv] == 0:
            queue.append((gv, 0))  # (node, level)
    
    while queue:
        gv, level = queue.popleft()
        level_map[level].append(gv)
        for neighbor in graph[gv]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append((neighbor, level+1))
    
    return level_map  # Level -> list of GVs at that level

def mf_handler(S=[],n=0,V=[],F=[],P={'0':[]},H=[]):
    # this loads env variables from .env file
    load_dotenv()
    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    dbname = os.getenv('DBNAME')
    # making connection to DB
    conn = psycopg2.connect("dbname="+dbname+" user="+user+" password="+password,
                            cursor_factory=psycopg2.extras.DictCursor)
    # getting cursor pointer to iterate over all rows
    cur = conn.cursor()
    cur.execute("SELECT * FROM sales")
    # creating dictionary, this is equivalent to MF table, but it is faster for level 0 gvs in dependency graph
    non_gv_dict = {}
    # this is basically the combined names of all columns in mf tables, it has cols V(attributes) and F(gv aggregations)
    VF = V+F

    # calling build_dependency to build dependency graph from predicates and then do topological sort to get level map
    graph,indegree = build_dependency_graph(P,V)
    level_map = level_order_toposort(graph, indegree)
    print(graph)
    # print(indegree)
    print(level_map)
    # iterate over all levels, level by level
    for level_key, level_values in level_map.items():
        # level key refers to the actual level, so 0th level has no dependency, this includes V and gv that are in MF format
        if level_key==0:
            for row in cur:
                row_key = ''
                row_value = []
                i = 0
                willAdd = True
                # P['0'] refers to conditions for V which are basically the where clause
                for condition in P['0']:
                    # checks condition vector for 0th variable
                    if recursive_condition(condition,0,len(condition)-1,row,toggle=0)==False:
                        willAdd = False
                if willAdd:
                    for v in V:  
                        # create key based on values of V, this will be used for lookup for updating MF table
                        row_key = row_key+str(row[v])
                        row_value.append(row[v])
                        i = i+1
                    j = i   
                    for f in F:
                        # get previous mf row to update or initialize empty values with none or 0 for count
                        if row_key in non_gv_dict and non_gv_dict[row_key][j] is not None:
                            row_value.append(non_gv_dict[row_key][j])
                        else:
                            if f"count_" in f:
                                row_value.append(0)
                            else:
                                row_value.append(None)
                        j=j+1
                    for f in F:
                        agg, gv, col = split_aggregation_string(f)
                        willUpdate_gv = True
                        # check condition for gv IF gv is part of current level_values
                        if gv in P:
                            for condition in P[gv]:
                                if gv in level_values:
                                    if recursive_condition(condition,0,len(condition)-1,row,toggle=0)==False:
                                        willUpdate_gv = False
                                else:
                                    willUpdate_gv = False
                        if willUpdate_gv:
                            # if conditions match then either update previous value or initialize new value
                            if row_value[i] is not None:
                                if row[col] is not None:
                                    row_value[i] = aggregation_handler(row_value[i],row[col],agg)
                            else:
                                if row[col] is not None:
                                    row_value[i] = aggregation_handler(None,row[col],agg,True)
                        i = i+1
                    # finally update dictionary MF row with updated row
                    non_gv_dict[row_key] = row_value
            # for levels>0, key is hard to use, so we convert dictionary into mf table
            # by this point the table is proper nxm table where n is number of rows and m number of columns
            non_gv_dict = dict(sorted(non_gv_dict.items()))
            mf_table = list(non_gv_dict.values())
        else:
            # fetch cursor pointer to start for every  level
            cur.execute("SELECT * FROM sales")
            
            for row in cur:
                willAdd = True
                # check base condition for 0th variable because this is where clause and happens before group by
                for condition in P['0']:
                    if recursive_condition(condition,0,len(condition)-1,row,toggle=0)==False:
                        willAdd = False
                if willAdd==False:
                    continue
                i = len(V)
                # for every gv, if gv is in current level, iterate through MF table and update rows for which condition is true
                for f in F:
                    agg, gv, col = split_aggregation_string(f)
                    if gv in level_values:
                        for values in mf_table:
                            willUpdate = True
                            for condition in P[gv]:
                                willUpdate = recursive_condition(condition,0,len(condition)-1,row,toggle=2,mf_val=values,combined_VF=VF)
                                if willUpdate==False:
                                    break
                            if willUpdate:
                                if values[i] is not None:
                                    if row[col] is not None:
                                        values[i] = aggregation_handler(values[i],row[col],agg)
                                else:
                                    if row[col] is not None:
                                        values[i] = aggregation_handler(None,row[col],agg,True)

                    i = i+1
                # break
    output = []
    output_headers = []
    #create final output table and output headers based on projection S 
    for value in mf_table:
        output_row = []
        if len(value)==0:
            continue
        for s in S:
            if len(output_headers)!=len(S):
                output_headers.append(s);
            output_row.append(parse_expression(s,value,VF)) 
        # check having condition and skip row if having condition is not met
        if len(H)!=0:
            if recursive_condition(H[0],0,len(H[0])-1,value,combined_VF=VF)==False:
                continue
        output.append(output_row)
    # tabulate final output
    output_table = tabulate.tabulate(output, headers=output_headers, tablefmt="psql")
    return output_table


S = ['cust','prod','count_ny_quant','avg_nyOrCt_quant','sum_ct_quant', 'sum_ct_quant/(count_ny_quant*(avg_nyOrCt_quant))']
P = {'0': ["('cust'='Sam' or 'cust'='Claire' or 'cust'='Helen') and ('prod'='Butter' or 'prod'='Dates')"],'nyOrCt':["('state'='NY' or 'state'='CT')"] ,'ny':["('state'='NY')"],'ct':["('ct.cust'<>'cust' and 'ct.prod'='prod' and 'state'='CT') and ('cust'='Helen' or 'cust'='Sam') and ('quant'>'1.2*avg_nyOrCt_quant')"]}
V = ['cust','prod']
F = ['count_ny_quant','avg_nyOrCt_quant','sum_ct_quant']
n = 3
H = ["((('count_ny_quant'>'avg_nyOrCt_quant' or 'count_ny_quant'<490) and ('sum_ct_quant'<'avg_nyOrCt_quant*3.3*count_ny_quant')))"]

dict_return = mf_handler(S=S,n=n,V=V, F=F,P=P,H=H)
print(dict_return)
with open("table_output.txt", "w") as file:
    file.write(dict_return)
