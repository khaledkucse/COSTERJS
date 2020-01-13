
const ts = require("typescript");

var root = "/home/khaledkucse/Project/backup/costerjs/";
var repos = root+"data/Repository/";
var dataset = root+"data/Dataset/";
var logfile = root+"dataExtract.log";


var surroundingTokenConsidered = 5;
var version = 0.1;
//the lexical tokens we will overlook while parsing
var removableLexicalKinds = [
    ts.SyntaxKind.EndOfFileToken,
    ts.SyntaxKind.NewLineTrivia,
    ts.SyntaxKind.WhitespaceTrivia
];

//the lexical tokens we will overlook while parsing
var removableContextToken = [
    ts.SyntaxKind.EndOfFileToken,
    ts.SyntaxKind.CommaToken,
    ts.SyntaxKind.ColonToken,
    ts.SyntaxKind.SemicolonToken,
    ts.SyntaxKind.DotToken,
    ts.SyntaxKind.OpenParenToken,
    ts.SyntaxKind.CloseParenToken,
    ts.SyntaxKind.FirstPunctuation,
    ts.SyntaxKind.CloseBraceToken,
    ts.SyntaxKind.OpenBracketToken,
    ts.SyntaxKind.CloseBracketToken
];

//the syntax templates
var templateKinds = [
    ts.SyntaxKind.TemplateHead,
    ts.SyntaxKind.TemplateMiddle,
    ts.SyntaxKind.TemplateSpan,
    ts.SyntaxKind.TemplateTail,
    ts.SyntaxKind.TemplateExpression,
    ts.SyntaxKind.TaggedTemplateExpression,
    ts.SyntaxKind.FirstTemplateToken,
    ts.SyntaxKind.LastTemplateToken,
    ts.SyntaxKind.TemplateMiddle
];

const keywords = ["async", "await", "break", "continue", "class", "extends", "constructor", "super","const",
                  "let", "var", "debugger", "delete", "do", "while", "export", "import", "for", "each", "in",
                  "of", "function", "return", "get", "set", "if", "else", "instanceof", "typeof", "null",
                  "undefined", "switch", "case", "default", "this", "true", "false", "try", "catch", "finally",
                  "void", "yield", "any", "boolean", "null", "never", "number", "string", "symbol", "undefined",
                  "as", "is", "enum", "type", "interface", "abstract", "implements","static", "readonly", "private",
                  "protected", "public", "declare", "module", "namespace", "require", "from", "of", "package"];



module.exports = {
    repos,
    dataset,
    logfile,
    removableLexicalKinds,
    removableContextToken,
    templateKinds,
    keywords,
    surroundingTokenConsidered,
    version
};