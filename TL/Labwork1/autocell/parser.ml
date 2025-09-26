type token =
  | EOF
  | DIMENSIONS
  | END
  | OF
  | ASSIGN
  | COMMA
  | LBRACKET
  | RBRACKET
  | DOT_DOT
  | DOT
  | PLUS
  | ID of (string)
  | INT of (int)

open Parsing;;
let _ = parse_error;;
# 17 "parser.mly"

open Common
open Ast
open Printf
open Symbols

(** Raise a syntax error with the given message.
	@param msg	Message of the error. *)
let error msg =
	raise (SyntaxError msg)


(** Restructure the when assignment into selections.
	@param f	Function to build the assignment.
	@param ws	Sequence of (expression, conditions) terminated
				by (expression, NO_COND).
	@return		Built statement. *)
let rec make_when f ws =
	match ws with
	| [(e, NO_COND)]	->	f e
	| (e, c)::t			-> IF_THEN(c, f e, make_when f t)
	| _ -> failwith "whens list not ended by (expression, NO_COND)."

# 43 "parser.ml"
let yytransl_const = [|
    0 (* EOF *);
  257 (* DIMENSIONS *);
  258 (* END *);
  259 (* OF *);
  260 (* ASSIGN *);
  261 (* COMMA *);
  262 (* LBRACKET *);
  263 (* RBRACKET *);
  264 (* DOT_DOT *);
  265 (* DOT *);
  266 (* PLUS *);
    0|]

let yytransl_block = [|
  267 (* ID *);
  268 (* INT *);
    0|]

let yylhs = "\255\255\
\001\000\002\000\002\000\004\000\004\000\005\000\003\000\003\000\
\006\000\006\000\007\000\008\000\008\000\008\000\008\000\000\000"

let yylen = "\002\000\
\007\000\003\000\001\000\001\000\003\000\005\000\000\000\001\000\
\003\000\003\000\005\000\001\000\001\000\001\000\003\000\002\000"

let yydefred = "\000\000\
\000\000\000\000\000\000\016\000\000\000\000\000\000\000\000\000\
\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\
\002\000\000\000\000\000\000\000\008\000\000\000\005\000\000\000\
\000\000\000\000\001\000\000\000\006\000\000\000\014\000\000\000\
\012\000\010\000\009\000\000\000\000\000\011\000\015\000"

let yydgoto = "\002\000\
\004\000\009\000\020\000\010\000\011\000\021\000\033\000\034\000"

let yysindex = "\007\000\
\245\254\000\000\008\255\000\000\007\255\248\254\009\255\003\255\
\011\255\010\255\000\000\002\255\004\255\252\254\006\255\012\255\
\000\000\013\255\014\255\019\000\000\000\017\255\000\000\015\255\
\018\255\250\254\000\000\250\254\000\000\016\255\000\000\019\255\
\000\000\000\000\000\000\023\255\020\255\000\000\000\000"

let yyrindex = "\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\022\255\000\000\000\000\000\000\022\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\026\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000"

let yygindex = "\000\000\
\000\000\000\000\000\000\000\000\016\000\000\000\020\000\005\000"

let yytablesize = 34
let yytable = "\018\000\
\003\000\018\000\007\000\008\000\031\000\032\000\019\000\001\000\
\005\000\006\000\013\000\012\000\014\000\016\000\015\000\017\000\
\007\000\026\000\027\000\024\000\028\000\007\000\030\000\003\000\
\025\000\013\000\029\000\036\000\037\000\038\000\023\000\039\000\
\035\000\022\000"

let yycheck = "\006\001\
\012\001\006\001\011\001\012\001\011\001\012\001\011\001\001\000\
\001\001\003\001\008\001\003\001\002\001\012\001\005\001\012\001\
\011\001\004\001\000\000\008\001\004\001\000\000\005\001\002\001\
\012\001\000\000\012\001\012\001\010\001\007\001\015\000\012\001\
\028\000\014\000"

let yynames_const = "\
  EOF\000\
  DIMENSIONS\000\
  END\000\
  OF\000\
  ASSIGN\000\
  COMMA\000\
  LBRACKET\000\
  RBRACKET\000\
  DOT_DOT\000\
  DOT\000\
  PLUS\000\
  "

let yynames_block = "\
  ID\000\
  INT\000\
  "

let yyact = [|
  (fun _ -> failwith "parser")
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 6 : int) in
    let _4 = (Parsing.peek_val __caml_parser_env 3 : 'config) in
    let _6 = (Parsing.peek_val __caml_parser_env 1 : 'opt_statements) in
    Obj.repr(
# 69 "parser.mly"
 (
		if _1 != 2 then error "only 2 dimension accepted";
		(_4, _6)
	)
# 144 "parser.ml"
               : Ast.prog))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : int) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : int) in
    Obj.repr(
# 77 "parser.mly"
  (
			if _1 >= _3 then error "illegal field values";
			[("", (0, (_1, _3)))]
		)
# 155 "parser.ml"
               : 'config))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'fields) in
    Obj.repr(
# 82 "parser.mly"
  ( set_fields _1 )
# 162 "parser.ml"
               : 'config))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'field) in
    Obj.repr(
# 87 "parser.mly"
  ( [_1] )
# 169 "parser.ml"
               : 'fields))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'fields) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'field) in
    Obj.repr(
# 89 "parser.mly"
  (_3 :: _1 )
# 177 "parser.ml"
               : 'fields))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 4 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 2 : int) in
    let _5 = (Parsing.peek_val __caml_parser_env 0 : int) in
    Obj.repr(
# 94 "parser.mly"
  (
			if _3 >= _5 then error "illegal field values";
			(_1, (_3, _5))
		)
# 189 "parser.ml"
               : 'field))
; (fun __caml_parser_env ->
    Obj.repr(
# 102 "parser.mly"
  ( NOP )
# 195 "parser.ml"
               : 'opt_statements))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'statement) in
    Obj.repr(
# 104 "parser.mly"
  ( _1 )
# 202 "parser.ml"
               : 'opt_statements))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'cell) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'expression) in
    Obj.repr(
# 110 "parser.mly"
  (
			if (fst _1) != 0 then error "assigned x must be 0";
			if (snd _1) != 0 then error "assigned Y must be 0";
			SET_CELL (0, _3)
		)
# 214 "parser.ml"
               : 'statement))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : 'expression) in
    Obj.repr(
# 116 "parser.mly"
        ( NOP )
# 222 "parser.ml"
               : 'statement))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 3 : int) in
    let _4 = (Parsing.peek_val __caml_parser_env 1 : int) in
    Obj.repr(
# 123 "parser.mly"
  (
			if (_2 < -1) || (_2 > 1) then error "x out of range";
			if (_4 < -1) || (_4 > 1) then error "x out of range";
			(_2, _4)
		)
# 234 "parser.ml"
               : 'cell))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'cell) in
    Obj.repr(
# 133 "parser.mly"
  ( CELL (0, fst _1, snd _1) )
# 241 "parser.ml"
               : 'expression))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : int) in
    Obj.repr(
# 135 "parser.mly"
  ( CST _1 )
# 248 "parser.ml"
               : 'expression))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 137 "parser.mly"
  ( NONE )
# 255 "parser.ml"
               : 'expression))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : int) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : int) in
    Obj.repr(
# 140 "parser.mly"
  ( NONE )
# 263 "parser.ml"
               : 'expression))
(* Entry program *)
; (fun __caml_parser_env -> raise (Parsing.YYexit (Parsing.peek_val __caml_parser_env 0)))
|]
let yytables =
  { Parsing.actions=yyact;
    Parsing.transl_const=yytransl_const;
    Parsing.transl_block=yytransl_block;
    Parsing.lhs=yylhs;
    Parsing.len=yylen;
    Parsing.defred=yydefred;
    Parsing.dgoto=yydgoto;
    Parsing.sindex=yysindex;
    Parsing.rindex=yyrindex;
    Parsing.gindex=yygindex;
    Parsing.tablesize=yytablesize;
    Parsing.table=yytable;
    Parsing.check=yycheck;
    Parsing.error_function=parse_error;
    Parsing.names_const=yynames_const;
    Parsing.names_block=yynames_block }
let program (lexfun : Lexing.lexbuf -> token) (lexbuf : Lexing.lexbuf) =
   (Parsing.yyparse yytables 1 lexfun lexbuf : Ast.prog)
