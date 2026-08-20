// Package cliutil holds output helpers shared by the bnk and bnk-server CLIs.
package cliutil

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
)

// Table writes an aligned table: a header row, a dashed rule under each
// column name, then the rows.
func Table(w io.Writer, headers []string, rows [][]string) {
	tw := tabwriter.NewWriter(w, 2, 4, 2, ' ', 0)
	fmt.Fprintln(tw, strings.Join(headers, "\t"))
	dashes := make([]string, len(headers))
	for i, h := range headers {
		dashes[i] = strings.Repeat("-", len(h))
	}
	fmt.Fprintln(tw, strings.Join(dashes, "\t"))
	for _, r := range rows {
		fmt.Fprintln(tw, strings.Join(r, "\t"))
	}
	tw.Flush()
}
