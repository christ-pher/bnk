// Package acl models the server-side access policy and compiles it into
// per-node inbound rules that clients enforce in their packet filter.
package acl

import (
	"fmt"
	"net/netip"
	"strconv"
	"strings"
)

// Policy is the admin-authored document: named groups of nodes and rules
// allowing traffic between selectors ("*", "group:<name>", or a node name).
type Policy struct {
	Groups map[string][]string `json:"groups,omitempty"`
	Rules  []Rule              `json:"rules"`
}

type Rule struct {
	From  []string `json:"from"`
	To    []string `json:"to"`
	Allow []string `json:"allow"` // "tcp/22", "udp/6000-6100", "icmp"
}

// NodeInfo is the slice of node state the compiler needs.
type NodeInfo struct {
	Name string
	IP   netip.Addr
	Tags []string
}

// CompiledRule is one inbound allowance for a destination node.
type CompiledRule struct {
	Srcs  []netip.Prefix `json:"srcs"`
	Proto string         `json:"proto"` // tcp | udp | icmp
	Ports []PortRange    `json:"ports,omitempty"`
}

type PortRange struct {
	First uint16 `json:"first"`
	Last  uint16 `json:"last"`
}

// Compile expands policy into inbound rules keyed by destination node name.
func Compile(p Policy, nodes []NodeInfo) (map[string][]CompiledRule, error) {
	byName := make(map[string]NodeInfo, len(nodes))
	for _, n := range nodes {
		byName[n.Name] = n
	}

	// expand resolves a selector list to a set of node names.
	expand := func(sels []string) (map[string]bool, error) {
		out := make(map[string]bool)
		for _, sel := range sels {
			switch {
			case sel == "*":
				for _, n := range nodes {
					out[n.Name] = true
				}
			case strings.HasPrefix(sel, "group:"):
				name := strings.TrimPrefix(sel, "group:")
				members, ok := p.Groups[name]
				if !ok {
					return nil, fmt.Errorf("acl: unknown group %q", name)
				}
				for _, m := range members {
					if _, ok := byName[m]; !ok {
						return nil, fmt.Errorf("acl: group %q member %q is not a known node", name, m)
					}
					out[m] = true
				}
			default:
				if _, ok := byName[sel]; !ok {
					return nil, fmt.Errorf("acl: %q is not a known node, group, or *", sel)
				}
				out[sel] = true
			}
		}
		return out, nil
	}

	compiled := make(map[string][]CompiledRule)
	for _, rule := range p.Rules {
		froms, err := expand(rule.From)
		if err != nil {
			return nil, err
		}
		tos, err := expand(rule.To)
		if err != nil {
			return nil, err
		}
		for dst := range tos {
			var srcs []netip.Prefix
			for src := range froms {
				if src == dst {
					continue
				}
				srcs = append(srcs, netip.PrefixFrom(byName[src].IP, 32))
			}
			if len(srcs) == 0 {
				continue
			}
			for _, allow := range rule.Allow {
				cr, err := parseAllow(allow)
				if err != nil {
					return nil, err
				}
				cr.Srcs = srcs
				compiled[dst] = append(compiled[dst], cr)
			}
		}
	}
	return compiled, nil
}

// parseAllow parses "tcp/22", "udp/6000-6100", or "icmp".
func parseAllow(s string) (CompiledRule, error) {
	proto, portSpec, hasPort := strings.Cut(s, "/")
	switch proto {
	case "icmp":
		if hasPort {
			return CompiledRule{}, fmt.Errorf("acl: icmp takes no port in %q", s)
		}
		return CompiledRule{Proto: "icmp"}, nil
	case "tcp", "udp":
		if !hasPort {
			return CompiledRule{}, fmt.Errorf("acl: %q needs a port like %s/22", s, proto)
		}
		first, last, isRange := strings.Cut(portSpec, "-")
		lo, err := strconv.ParseUint(first, 10, 16)
		if err != nil {
			return CompiledRule{}, fmt.Errorf("acl: bad port in %q: %w", s, err)
		}
		hi := lo
		if isRange {
			hi, err = strconv.ParseUint(last, 10, 16)
			if err != nil || hi < lo {
				return CompiledRule{}, fmt.Errorf("acl: bad port range in %q", s)
			}
		}
		return CompiledRule{Proto: proto, Ports: []PortRange{{uint16(lo), uint16(hi)}}}, nil
	default:
		return CompiledRule{}, fmt.Errorf("acl: unknown protocol in %q (want tcp, udp, or icmp)", s)
	}
}

// Match reports whether a packet from src with proto/dstPort is allowed by
// this rule. Port is ignored for icmp.
func (r CompiledRule) Match(src netip.Addr, proto string, dstPort uint16) bool {
	if proto != r.Proto {
		return false
	}
	var inSrc bool
	for _, p := range r.Srcs {
		if p.Contains(src) {
			inSrc = true
			break
		}
	}
	if !inSrc {
		return false
	}
	if r.Proto == "icmp" {
		return true
	}
	for _, pr := range r.Ports {
		if dstPort >= pr.First && dstPort <= pr.Last {
			return true
		}
	}
	return false
}
