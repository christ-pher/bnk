package cliutil_test

import (
	"bytes"
	"testing"

	"vpnmesh/internal/cliutil"
)

func TestTableDashesUnderHeaders(t *testing.T) {
	var buf bytes.Buffer
	cliutil.Table(&buf, []string{"PEER", "IP", "ONLINE"}, [][]string{
		{"alpha*", "100.64.0.1", "true"},
		{"beta", "100.64.0.2", "false"},
	})
	want := "" +
		"PEER    IP          ONLINE\n" +
		"----    --          ------\n" +
		"alpha*  100.64.0.1  true\n" +
		"beta    100.64.0.2  false\n"
	if got := buf.String(); got != want {
		t.Errorf("table output:\n%q\nwant:\n%q", got, want)
	}
}

func TestTableNoRows(t *testing.T) {
	var buf bytes.Buffer
	cliutil.Table(&buf, []string{"ID", "NAME"}, nil)
	want := "ID  NAME\n--  ----\n"
	if got := buf.String(); got != want {
		t.Errorf("table output:\n%q\nwant:\n%q", got, want)
	}
}
