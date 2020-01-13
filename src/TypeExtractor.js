"use strict";

//Importing required module
const config = require("./config");
const ts = require("typescript");
const fs = require("fs");
const path = require("path");

//Print function redesigned
function print(x) { console.log(x); }


/*
This Function visit each project and parse all js or ts files
@param dirPath: Path of the directory or project(String)
@param SuccessCounter: Number of projects successfully parsed
@param errorCounter: Number of projects create error while parsing
 */
function visitProject(dirPath,successCounter,filesCounter,dataCounter,errorCounter) {
    let dirContent = fs.readdirSync(dirPath);
    //When the tsconfog.jason file means the typescript configuration file is present at the project
    if (dirContent.find(value => value === "tsconfig.json")) {
        successCounter=successCounter+1;

        // Extract the sequence of the js files
        let values = extractContextAndLabel(dirPath,filesCounter,dataCounter);
        filesCounter = values[0];
        dataCounter = values[1];
    }
    else {
        //if tsconfig.json file not present then for each file in the project we search the js files
        dirContent.forEach(function (eachFile) {
            let filePath = dirPath + "/" + eachFile;
            try {
                if (fs.statSync(filePath).isDirectory()){
                    if(filePath.indexOf("DefinitelyTyped") < 0 && filePath.indexOf("TypeScript/tests") < 0 && eachFile !== ".git") {
                        visitProject(filePath);
                    }
                }
            }
            catch (err) {
                errorCounter = errorCounter + 1;
            }
        });
    }
    return [successCounter, filesCounter,dataCounter,errorCounter];
}



/*
Collects js files, create compilation unit from these files and extract the script sequence for each file
@param dirPath: path of the project
 */
function extractContextAndLabel(dirPath,filesCounter,dataCounter) {

    //Step 1: Walk through the project directory and collect the path of java script files
    let jsFiles = [];
    collectJSFiles(dirPath, jsFiles);
    fs.appendFileSync(config.logfile,'Project: '+dirPath+'\n','utf-8');
    fs.appendFileSync(config.logfile,'Number of Java Script Files found: '+jsFiles.length+'\n','utf-8');

    //Step 2: Create program instance for each script file collected in Step 1 and get the types
    let program = ts.createProgram(jsFiles, {target: ts.ScriptTarget.Latest,
                                                    module: ts.ModuleKind.CommonJS,
                                                    checkJs: true,
                                                    allowJs: true });

    let checker = null;
    try {
        checker = program.getTypeChecker();
    }
    catch (err) {
        return null;
    }

    fs.appendFileSync(config.logfile,'Number of Compilation units: '+program.getSourceFiles().length+'\n','utf-8');


    //Step 3: For each source file visit and extract the types of tokens
    for (let eachSourceFile of program.getSourceFiles()) {
        let filename = eachSourceFile.getSourceFile().fileName;
        if (filename.endsWith('.d.ts'))
            continue;
        try {
            let relativePath = path.relative(dirPath, filename);
            if (relativePath.startsWith(".."))
                continue;
            let tokens=[], contexts = [],labels = [],tokenPos=[],stmtPos = [],resolvedTypes=[],postTypeStarts=[];
            let values = visitSourceFile(eachSourceFile, checker, filename, tokens, contexts, labels,stmtPos,tokenPos,resolvedTypes,postTypeStarts);
            tokens = values[0];
            contexts = values[1];
            labels = values[2];
            stmtPos = values[3];
            tokenPos = values[4];
            resolvedTypes=values[5];
            postTypeStarts=values[6];
            let finaldataset = [];
            for(let index=0;index<contexts.length;index++) {
                //collecting post tokens of the token in the context
                for (let i = postTypeStarts[index];i<(postTypeStarts[index]+config.surroundingTokenConsidered);i++){
                    if(i >= resolvedTypes.length)
                        break;
                    contexts[index].push(resolvedTypes[i]);
                }
                //store the extracted sequence in the final dataset.
                if( contexts[index].length >= 5){
                    finaldataset.push([filename.replace(config.repos,"").replace(new RegExp(',', 'g'),''), stmtPos[index].replace(new RegExp(',', 'g'),''),tokens[index].replace(new RegExp(',', 'g'),'comma'),tokenPos[index],contexts[index].toString().replace(new RegExp(',', 'g'),' '), labels[index].replace(new RegExp(',', 'g'),'')]);
                }
            }
            if(finaldataset.length > 0){
                let filePath = filename.replace(config.repos,"");
                let filePathToken = filePath.split("/");
                let fileName = filePathToken.pop();
                filePath = config.dataset+filePathToken.join("/")+"/"+fileName.substr(0,fileName.length-2)+"csv";
                fs.mkdirSync(path.dirname(filePath),{recursive:true});
                fs.writeFileSync(filePath, finaldataset.join("\n"), 'utf-8');
            }
            dataCounter = dataCounter + finaldataset.length;
            filesCounter = filesCounter + 1;
        }
        catch (e) {}
    }

    return [filesCounter,dataCounter];
}

/*
The function collects all JS files form a directory.
@param director: The path of the directory/project
@param filelist: A list of paths of current js files(Initially empty)
@return the list of path of js files
 */
function collectJSFiles(directory, filelist) {
    var fs = fs || require('fs'), files = fs.readdirSync(directory);
    filelist = filelist || [];
    files.forEach(function (eachFile) {
        let fullPath = path.join(directory, eachFile);
        try {
            if (fs.statSync(fullPath).isDirectory()) {
                if (eachFile !== ".git")
                    filelist = collectJSFiles(directory + '/' + eachFile, filelist);
            }
            else if ((eachFile.endsWith('.js') || eachFile.endsWith('.ts')) && fs.statSync(fullPath).size < 1000 * 1000)
                filelist.push(fullPath);
        }
        catch (e) {}
    });
    return filelist;
}

/*

 */
function visitSourceFile(sourceFile, checker, filePath, tokens,contexts, labels,stmtPos,tokenPos,resolvedTypes,postTypeStarts) {
    //for each node in the AST
    for (let index in sourceFile.getChildren()) {
        let indexnum = parseInt(index);
        let child = sourceFile.getChildren()[indexnum];
        //Remove the lexical tokens descrived at removableLexicalKinds
        if (config.removableLexicalKinds.indexOf(child.kind) !== -1 ||
            ts.SyntaxKind[child.kind].indexOf("JSDoc") !== -1) {
            continue;
        }
        // Tentatively remove all templates as these substantially hinder type/token alignment; to be improved in the future
        else if (config.templateKinds.indexOf(child.kind) !== -1) {
            continue;
        }
        //when the childnode is the leaf node
        if (child.getChildCount() === 0) {
            //fs.appendFileSync(config.logfile,child.getText()+":"+ts.SyntaxKind[child.kind]+"\n","utf-8");
            //fs.appendFileSync(config.logfile,"===========\n","utf-8");
            let eachContext=[];
            let eachLabel = "";
            let eachStmtPos = ""+child.pos+":"+child.end+"";
            switch (child.kind) {
                //When the type is from identifier
                case ts.SyntaxKind.Identifier:
                    try {
                        //Bind the type of the identifier
                        let symbol = checker.getSymbolAtLocation(child);

                        //When the type binding return not null, means the type is resolved successfully
                        if (symbol) {
                            //collect the type of the identifier
                            let type = checker.typeToString(checker.getTypeOfSymbolAtLocation(symbol, child));
                            //Avoid if the identifier is unknown, type of another identifier, has numerical or string in its name
                            if (checker.isUnknownSymbol(symbol) || type.startsWith("typeof") || type.startsWith("\"")
                                || type.match("[0-9]+") || type === "any") {
                                switch (child.parent.kind) {
                                    case ts.SyntaxKind.ElementAccessExpression:
                                        eachLabel = "ElementAccessExpression";
                                        resolvedTypes.push('ElementAccessExpression');
                                        break;
                                    case ts.SyntaxKind.PropertyAccessExpression:
                                        eachLabel = "PropertyAccessExpression";
                                        resolvedTypes.push('PropertyAccessExpression');
                                        break;
                                    case ts.SyntaxKind.MethodDeclaration:
                                        eachLabel = "MethodDeclaration";
                                        resolvedTypes.push('MethodDeclaration');
                                        break;
                                    case ts.SyntaxKind.CallExpression:
                                        eachLabel = "CallExpression";
                                        resolvedTypes.push('CallExpression');
                                        break;
                                    case ts.SyntaxKind.ReturnStatement:
                                        eachLabel = "ReturnStatement";
                                        resolvedTypes.push('ReturnStatement');
                                        break;
                                    case ts.SyntaxKind.Parameter:
                                        eachLabel = "Parameter";
                                        resolvedTypes.push('Parameter');
                                        break;
                                    default:
                                        eachLabel = "any";
                                        resolvedTypes.push('Identifier');
                                        break;
                                }
                            } else {
                                eachLabel = "" + type + "";
                                resolvedTypes.push(type.toString());
                            }
                            break;
                        } else {
                            resolvedTypes.push('Identifier');
                        }
                    } catch (e) {}
                    break;
                case ts.SyntaxKind.NumericLiteral:
                    eachLabel = "NumericLiteral";
                    resolvedTypes.push('NumericLiteral');
                    break;
                case ts.SyntaxKind.BigIntLiteral:
                    eachLabel = "BigIntLiteral";
                    resolvedTypes.push('BigIntLiteral');
                    break;
                case ts.SyntaxKind.StringLiteral:
                    eachLabel = "StringLiteral";
                    resolvedTypes.push('StringLiteral');
                    break;
                case ts.SyntaxKind.RegularExpressionLiteral:
                    try {
                        let symbol = checker.getSymbolAtLocation(child);
                        if (symbol) {
                            let type = checker.typeToString(checker.getTypeOfSymbolAtLocation(symbol, child));
                            eachLabel = ""+type+"";
                            resolvedTypes.push(type.toString());
                        }
                        else{
                            resolvedTypes.push('RegularExpressionLiteral');
                        }
                        break;
                    }
                    catch (e) {}
                    break;
                case ts.SyntaxKind.ThisKeyword:
                    eachLabel = "ThisKeyword";
                    resolvedTypes.push('ThisKeyword');
                    break;
                default:
                    if(config.removableContextToken.includes(child.kind)){}
                    else{
                        resolvedTypes.push(ts.SyntaxKind[child.kind].toString());
                    }
                    break;
            }
            eachLabel = eachLabel.trim();
            //When the label is too long with the type information at last
            if (eachLabel.match(".+ => .+")) {
                eachLabel = "" + eachLabel.substring(eachLabel.lastIndexOf(" => ") + 4);
                resolvedTypes.pop();
                resolvedTypes.push(eachLabel);
            }

            //if the type is unknown or not found
            if(eachLabel === "" || child.getText().trim() === "")
                continue;

            //When the code has too complex element to parse
            if (eachLabel.match("\\s")) {
                eachLabel = "complex";
                resolvedTypes.pop();
                resolvedTypes.push(eachLabel);
            }
            //if the SyntaxKind is template type. Those have nothing to resolve.
            if (eachLabel !== "") {
                let parentKind = ts.SyntaxKind[sourceFile.kind];
                if (parentKind.toLowerCase().indexOf("template") >= 0)
                    continue;
            }
            //Collecting previous tokens in the context
            for (let i = resolvedTypes.length-2;i>=0;i--)
            {
                if(eachContext.length > config.surroundingTokenConsidered)
                    break;
                eachContext.push(resolvedTypes[i]);
            }
            eachContext = eachContext.reverse();

            //storing actual code tokens, contexts, actual labels, position of statement, position of token in the context
            tokens.push(child.getText().replace(new RegExp(',', 'g'),'').trim());
            contexts.push(eachContext);
            labels.push(eachLabel.replace(new RegExp(',', 'g'),'').trim());
            stmtPos.push(eachStmtPos);
            tokenPos.push(eachContext.length);
            postTypeStarts.push(resolvedTypes.length);
        }
        //If the node in the AST is not the leaf node
        else {
            visitSourceFile(child, checker,filePath, tokens, contexts, labels,stmtPos,tokenPos,resolvedTypes,postTypeStarts);
        }
    }
    return [tokens,contexts,labels,stmtPos,tokenPos,resolvedTypes,postTypeStarts];
}


let successCounter = 0, fileCounter = 0, dataCounter = 0,errorCounter = 0;
print('COSTERJS: A Fast and Scalable Java Script Type Infer Tool');
print('Version:'+config.version);
print('\nData Extraction Part:\n=============================================================================================');
print('To see the more information for each project open '+config.logfile+'\n\n\n');
print('Starting to extract data from '+config.repos);
try {
    if (fs.existsSync(config.logfile)) {
        fs.unlinkSync(config.logfile);
    }
} catch(err) {
    console.error(err);
}
//Read each eachrepository
for (let eachrepository of fs.readdirSync(config.repos)) {
    for (let eachproject of fs.readdirSync(config.repos + eachrepository)) {
        
        // This project stalls forever
        if (eachrepository === "SAP") continue;

        //Collecting directory path
        let dirPath = config.repos +  eachrepository + "/" + eachproject;
        fs.appendFileSync(config.logfile, "Repository: "+dirPath+"\n", 'utf-8');
        //visiting each directory. This is the entry point of parsing
        let counter = visitProject(dirPath,successCounter,fileCounter,dataCounter,errorCounter);
        successCounter = counter[0];
        fileCounter = counter[1];
        dataCounter = counter[2];
        errorCounter = counter[3];
        fs.appendFileSync(config.logfile, "=============================================================================================\n", 'utf-8');
    }
}
print('Total Number of projects: '+successCounter);
fs.appendFileSync(config.logfile, 'Total Number of projects: '+successCounter+'\n', 'utf-8');
print('Total Number of files: '+fileCounter);
fs.appendFileSync(config.logfile, 'Total Number of files: '+fileCounter+'\n', 'utf-8');
print('Total Number of data: '+dataCounter);
fs.appendFileSync(config.logfile, 'Total Number of data: '+dataCounter+'\n', 'utf-8');
print('Total Number of files found erroneous: '+errorCounter);
fs.appendFileSync(config.logfile, 'Total Number of files found erroneous: '+errorCounter+'\n', 'utf-8');


