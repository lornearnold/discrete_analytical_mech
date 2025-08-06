Sub ChangeEquationFontToXITS()
    Dim eq As OMath
    For Each eq In ActiveDocument.OMaths
        eq.Range.Font.Name = "XITS Math"
    Next eq
End Sub