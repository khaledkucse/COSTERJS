package org.srlab.costerjs;

public class CodeToken {
    private String filePath;
    private String stmtPos;
    private String codeToken;
    private String tokenPos;
    private String context;
    private String actualLabel;

    public CodeToken() {
    }

    public CodeToken(String filePath, String stmtPos, String codeToken, String tokenPos, String context, String actualLabel) {
        this.filePath = filePath;
        this.stmtPos = stmtPos;
        this.codeToken = codeToken;
        this.tokenPos = tokenPos;
        this.context = context;
        this.actualLabel = actualLabel;
    }

    public String getFilePath() {
        return filePath;
    }

    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }

    public String getStmtPos() {
        return stmtPos;
    }

    public void setStmtPos(String stmtPos) {
        this.stmtPos = stmtPos;
    }

    public String getCodeToken() {
        return codeToken;
    }

    public void setCodeToken(String codeToken) {
        this.codeToken = codeToken;
    }

    public String getTokenPos() {
        return tokenPos;
    }

    public void setTokenPos(String tokenPos) {
        this.tokenPos = tokenPos;
    }

    public String getContext() {
        return context;
    }

    public void setContext(String context) {
        this.context = context;
    }

    public String getActualLabel() {
        return actualLabel;
    }

    public void setActualLabel(String actualLabel) {
        this.actualLabel = actualLabel;
    }
}
